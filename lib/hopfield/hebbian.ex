defmodule Hopfield.Hebbian do
  @moduledoc """
  Hebbian learning for bipolar Hopfield memories.
  """

  @type activation :: -1 | 1
  @type state :: [activation()]
  @type index :: non_neg_integer()
  @type weight_matrix :: %{index() => %{index() => float()}}

  @doc """
  Builds the Hopfield weight matrix from stored memories.

  This is the outer-product memory rule used in the traditional Hopfield model:
  memories that have matching signs strengthen a connection, and memories with
  opposite signs weaken it. The zero diagonal removes self-coupling.

  References: Hebb 1949 for the learning postulate; Hopfield 1982 and
  Amit, Gutfreund, and Sompolinsky 1985 for Hopfield associative memory.

  Formula:

      w_ij = sum(memory_i * memory_j) / n
      w_ii = 0
  """
  @spec weight_matrix([state()]) :: weight_matrix()
  def weight_matrix(memories) do
    size = validate_states!(memories, "memory")
    indices = 0..(size - 1)

    for i <- indices, into: %{} do
      row =
        for j <- indices, into: %{} do
          # Hebbian outer product summed over memories:
          # w_ij = (1 / n) * sum_mu memory_mu[i] * memory_mu[j].
          {j, weight(memories, i, j, size)}
        end

      {i, row}
    end
  end

  @doc false
  @spec validate_states!([state()], String.t()) :: pos_integer()
  def validate_states!([], label), do: raise(ArgumentError, "at least one #{label} is required")

  def validate_states!(states, label) when is_list(states) do
    size =
      states
      |> hd()
      |> validate_state!("#{label} 0")
      |> length()

    states
    |> Enum.with_index()
    |> Enum.each(fn {state, index} ->
      length = state |> validate_state!("#{label} #{index}") |> length()

      if length != size do
        raise ArgumentError,
              "all #{label}s must have the same length; #{label} #{index} has length #{length}, expected #{size}"
      end
    end)

    size
  end

  defp weight(_memories, same, same, _size), do: 0.0

  defp weight(memories, i, j, size) do
    # Same-sign activations contribute positive coupling; opposite signs
    # contribute negative coupling.
    memories
    |> Enum.reduce(0, fn memory, acc ->
      acc + Enum.at(memory, i) * Enum.at(memory, j)
    end)
    |> Kernel./(size)
  end

  defp validate_state!(state, label) when is_list(state) and state != [] do
    if Enum.all?(state, &(&1 in [-1, 1])) do
      state
    else
      raise ArgumentError, "#{label} must contain only bipolar activations, -1 or 1"
    end
  end

  defp validate_state!(_state, label) do
    raise ArgumentError, "#{label} must be a non-empty list"
  end
end
