defmodule Hopfield.Network do
  @moduledoc """
  Coordinates a process-per-neuron Hopfield network.

  This module is intentionally plain: the network is a struct containing a
  weight matrix and neuron pids. Only neurons are processes.
  """

  alias Hopfield.{Hebbian, Neuron}

  defstruct [:size, :weight_matrix, neurons: %{}]

  @type t :: %__MODULE__{
          size: pos_integer(),
          weight_matrix: Hebbian.weight_matrix(),
          neurons: %{non_neg_integer() => pid()}
        }

  @type recall_result :: %{
          state: [Neuron.activation()],
          fixed_point?: boolean(),
          sweeps: non_neg_integer(),
          energy: float()
        }

  @spec new(Hebbian.weight_matrix(), [Neuron.activation()]) :: t()
  def new(weight_matrix, initial_state) do
    size = validate_state!(initial_state, weight_matrix)

    # One neuron process per component of the network state s.
    neurons =
      initial_state
      |> Enum.with_index()
      |> Enum.map(fn {activation, index} ->
        pid =
          Neuron.spawn(
            index,
            activation,
            Map.fetch!(weight_matrix, index)
          )

        {index, pid}
      end)
      |> Map.new()

    network = %__MODULE__{size: size, weight_matrix: weight_matrix, neurons: neurons}

    # Fully connected Hopfield graph without self-connections.
    Enum.each(neurons, fn {index, neuron} ->
      peers = Map.delete(neurons, index)
      :ok = Neuron.connect(neuron, peers)
    end)

    network
  end

  @spec stop(t()) :: :ok
  def stop(%__MODULE__{} = network) do
    Enum.each(network.neurons, fn {_index, neuron} ->
      if Process.alive?(neuron) do
        Neuron.stop(neuron)
      end
    end)
  end

  @spec state(t()) :: [Neuron.activation()]
  def state(%__MODULE__{} = network), do: current_state(network)

  @spec set_state(t(), [Neuron.activation()]) :: :ok
  def set_state(%__MODULE__{} = network, state) do
    validate_state!(state, network.weight_matrix)

    state
    |> Enum.with_index()
    |> Enum.each(fn {activation, index} ->
      network.neurons
      |> Map.fetch!(index)
      |> Neuron.set_activation(activation)
    end)
  end

  @spec asynchronous_sweep(t()) :: [Neuron.update_result()]
  def asynchronous_sweep(%__MODULE__{} = network) do
    # Hopfield's discrete dynamics update neurons asynchronously. Here a sweep is
    # deterministic and simple: neuron 0, then 1, ..., then n - 1.
    run_asynchronous_sweep(network)
  end

  @spec settle(t(), non_neg_integer()) :: recall_result()
  def settle(%__MODULE__{} = network, max_sweeps \\ 20) do
    max_sweeps = validate_max_sweeps!(max_sweeps)
    {sweeps, fixed_point?} = settle_until_fixed_point(network, max_sweeps, 0)

    %{
      state: current_state(network),
      fixed_point?: fixed_point?,
      sweeps: sweeps,
      energy: energy(network)
    }
  end

  @spec energy(t()) :: float()
  def energy(%__MODULE__{} = network) do
    # Hopfield energy for symmetric weights:
    # E(s) = -1/2 * sum_i sum_j w_ij * s_i * s_j.
    # With asynchronous sign updates this is the Lyapunov function described by
    # Hopfield 1982.
    activations =
      network
      |> current_state()
      |> Enum.with_index()
      |> Map.new(fn {activation, index} -> {index, activation} end)

    total =
      Enum.reduce(0..(network.size - 1), 0.0, fn i, acc ->
        Enum.reduce(0..(network.size - 1), acc, fn j, inner_acc ->
          inner_acc + Map.fetch!(network.weight_matrix[i], j) * activations[i] * activations[j]
        end)
      end)

    -0.5 * total
  end

  defp settle_until_fixed_point(_network, 0, sweeps), do: {sweeps, false}

  defp settle_until_fixed_point(network, remaining, sweeps) do
    updates = run_asynchronous_sweep(network)
    sweeps = sweeps + 1

    if Enum.any?(updates, & &1.changed?) do
      settle_until_fixed_point(network, remaining - 1, sweeps)
    else
      {sweeps, true}
    end
  end

  defp run_asynchronous_sweep(network) do
    Enum.map(0..(network.size - 1), fn index ->
      network.neurons
      |> Map.fetch!(index)
      |> Neuron.update()
    end)
  end

  defp current_state(network) do
    network.neurons
    |> Enum.sort_by(fn {index, _neuron} -> index end)
    |> Enum.map(fn {_index, neuron} -> Neuron.activation(neuron) end)
  end

  defp validate_state!(state, weight_matrix) do
    size = Hebbian.validate_states!([state], "state")

    if map_size(weight_matrix) != size do
      raise ArgumentError,
            "state has length #{size}, expected #{map_size(weight_matrix)}"
    end

    size
  end

  defp validate_max_sweeps!(max_sweeps) when is_integer(max_sweeps) and max_sweeps >= 0 do
    max_sweeps
  end

  defp validate_max_sweeps!(_max_sweeps) do
    raise ArgumentError, "max_sweeps must be a non-negative integer"
  end
end
