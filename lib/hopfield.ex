defmodule Hopfield do
  @moduledoc """
  A small, traditional Hopfield network where every neuron is an Elixir process.

  Memories and states are represented as lists of bipolar activations:
  `1` for on and `-1` for off.
  """

  alias Hopfield.{Hebbian, Network}

  @type activation :: -1 | 1
  @type state :: [activation()]
  @type memory :: state()

  @doc """
  Builds a Hopfield weight matrix from stored memories using Hebbian learning.

  References: Hopfield 1982; Amit, Gutfreund, and Sompolinsky 1985.

  ## Examples

      iex> weights = Hopfield.hebbian_weights([[1, -1, 1]])
      iex> weights[0][0]
      0.0
      iex> weights[0][1]
      -0.3333333333333333
  """
  @spec hebbian_weights([memory()]) :: Hebbian.weight_matrix()
  def hebbian_weights(memories), do: Hebbian.weight_matrix(memories)

  @doc """
  Builds a process-backed Hopfield network.
  """
  @spec new([memory()]) :: Network.t()
  def new(memories), do: new(memories, hd(memories))

  @doc """
  Builds a process-backed Hopfield network with an explicit initial state.
  """
  @spec new([memory()], state()) :: Network.t()
  def new(memories, initial_state) do
    memories
    |> hebbian_weights()
    |> Network.new(initial_state)
  end

  @doc """
  Recalls the closest attractor from a cue state.

  This builds a temporary process-backed network, lets it settle to a fixed
  point, and then stops the neuron processes.
  """
  @spec recall([memory()], state(), non_neg_integer()) :: Network.recall_result()
  def recall(memories, cue, max_sweeps \\ 20) do
    network = new(memories, cue)

    try do
      Network.settle(network, max_sweeps)
    after
      Network.stop(network)
    end
  end
end
