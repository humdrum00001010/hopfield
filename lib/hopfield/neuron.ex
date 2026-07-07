defmodule Hopfield.Neuron do
  @moduledoc """
  A single Hopfield neuron implemented as a plain Elixir process.

  The process keeps its own activation in a recursive `receive` loop. During a
  update, it sends messages to peer neuron processes, waits for their activation
  replies, computes the local field, and then recurses with the new state.
  """

  import Kernel, except: [spawn: 1]

  defstruct [:index, :activation, :weights, peers: %{}]

  @type activation :: -1 | 1
  @type update_result :: %{
          index: non_neg_integer(),
          previous_activation: activation(),
          activation: activation(),
          local_field: float(),
          changed?: boolean()
        }

  @spec spawn(non_neg_integer(), activation(), %{non_neg_integer() => float()}) :: pid()
  def spawn(index, activation, weights) do
    neuron = %__MODULE__{
      index: index,
      activation: activation,
      weights: weights
    }

    Kernel.spawn(fn -> loop(neuron) end)
  end

  @spec connect(pid(), %{non_neg_integer() => pid()}) :: :ok
  def connect(neuron, peers) when is_pid(neuron) and is_map(peers) do
    call(neuron, {:connect, peers})
  end

  @spec activation(pid()) :: activation()
  def activation(neuron) when is_pid(neuron) do
    call(neuron, :activation)
  end

  @spec set_activation(pid(), activation()) :: :ok
  def set_activation(neuron, activation) when is_pid(neuron) and activation in [-1, 1] do
    call(neuron, {:set_activation, activation})
  end

  @spec update(pid()) :: update_result()
  def update(neuron) when is_pid(neuron) do
    # One asynchronous Hopfield update for this neuron.
    call(neuron, :update)
  end

  @spec stop(pid()) :: :ok
  def stop(neuron) when is_pid(neuron) do
    call(neuron, :stop)
  end

  defp loop(neuron) do
    receive do
      {caller, ref, {:connect, peers}} ->
        send(caller, {ref, :ok})
        loop(%{neuron | peers: peers})

      {caller, ref, :activation} ->
        send(caller, {ref, neuron.activation})
        loop(neuron)

      {caller, ref, {:set_activation, activation}} ->
        send(caller, {ref, :ok})
        loop(%{neuron | activation: activation})

      {caller, ref, :update} ->
        {reply, next_neuron} = update_neuron(neuron)
        send(caller, {ref, reply})
        loop(next_neuron)

      {caller, ref, :stop} ->
        send(caller, {ref, :ok})

      {:activation_request, requester, ref} ->
        send(requester, {:activation_reply, ref, neuron.index, neuron.activation})
        loop(neuron)
    end
  end

  defp update_neuron(%{peers: peers} = neuron) when map_size(peers) == 0 do
    {result(neuron, 0.0, neuron.activation), neuron}
  end

  defp update_neuron(neuron) do
    ref = make_ref()
    expected = MapSet.new(Map.keys(neuron.peers))

    # Ask peer neuron processes for the current state components s_j.
    Enum.each(neuron.peers, fn {_index, peer} ->
      send(peer, {:activation_request, self(), ref})
    end)

    activations = collect_activation_replies(ref, expected, %{}, neuron)

    # Local field h_i = sum_j w_ij * s_j, followed by s_i <- sign(h_i).
    local_field = local_field(activations, neuron.weights)
    next_activation = sign(local_field, neuron.activation)

    {result(neuron, local_field, next_activation), %{neuron | activation: next_activation}}
  end

  defp collect_activation_replies(ref, expected, activations, neuron) do
    if map_size(activations) == MapSet.size(expected) do
      activations
    else
      receive do
        {:activation_reply, ^ref, index, activation} ->
          activations =
            if MapSet.member?(expected, index) do
              Map.put(activations, index, activation)
            else
              activations
            end

          collect_activation_replies(ref, expected, activations, neuron)

        {:activation_request, requester, other_ref} ->
          send(requester, {:activation_reply, other_ref, neuron.index, neuron.activation})
          collect_activation_replies(ref, expected, activations, neuron)

        {caller, caller_ref, :activation} ->
          send(caller, {caller_ref, neuron.activation})
          collect_activation_replies(ref, expected, activations, neuron)

        {caller, caller_ref, _message} ->
          send(caller, {caller_ref, {:error, :updating}})
          collect_activation_replies(ref, expected, activations, neuron)
      end
    end
  end

  defp call(pid, message, timeout \\ 5_000) do
    ref = make_ref()

    # Minimal synchronous request/reply protocol built from send/receive.
    send(pid, {self(), ref, message})

    receive do
      {^ref, reply} -> reply
    after
      timeout -> exit({:timeout, {__MODULE__, message}})
    end
  end

  defp local_field(activations, weights) do
    Enum.reduce(activations, 0.0, fn {index, activation}, acc ->
      acc + Map.fetch!(weights, index) * activation
    end)
  end

  defp sign(local_field, _previous) when local_field > 0.0, do: 1
  defp sign(local_field, _previous) when local_field < 0.0, do: -1
  defp sign(_local_field, previous), do: previous

  defp result(neuron, local_field, next_activation) do
    %{
      index: neuron.index,
      previous_activation: neuron.activation,
      activation: next_activation,
      local_field: local_field,
      changed?: next_activation != neuron.activation
    }
  end
end
