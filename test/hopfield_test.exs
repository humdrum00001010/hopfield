defmodule HopfieldTest do
  use ExUnit.Case

  doctest Hopfield

  alias Hopfield.{Network, Neuron}

  test "learns symmetric Hebbian weights with a zero diagonal" do
    weights = Hopfield.hebbian_weights([[1, -1, 1], [-1, 1, -1]])

    assert weights[0][0] == 0.0
    assert weights[1][1] == 0.0
    assert weights[2][2] == 0.0

    assert weights[0][1] == weights[1][0]
    assert weights[0][2] == weights[2][0]
    assert weights[0][1] < 0.0
    assert weights[0][2] > 0.0
  end

  test "recalls a stored pattern from a noisy cue" do
    memory = [1, -1, 1, -1, 1]
    cue = [1, -1, -1, -1, 1]

    result = Hopfield.recall([memory], cue)

    assert result.fixed_point?
    assert result.state == memory
    assert result.sweeps <= 3
  end

  test "network performs asynchronous sweeps through neuron processes" do
    weight_matrix = Hopfield.hebbian_weights([[1, -1, 1, -1]])
    network = Network.new(weight_matrix, [1, -1, -1, -1])

    on_exit(fn ->
      Network.stop(network)
    end)

    before_energy = Network.energy(network)
    updates = Network.asynchronous_sweep(network)
    after_energy = Network.energy(network)

    assert Enum.any?(updates, & &1.changed?)
    assert Network.state(network) == [1, -1, 1, -1]
    assert after_energy <= before_energy
  end

  test "a neuron samples peer process activations before updating" do
    left = Neuron.spawn(0, -1, %{1 => 1.0})
    right = Neuron.spawn(1, 1, %{0 => 1.0})

    on_exit(fn ->
      for neuron <- [left, right], Process.alive?(neuron), do: Neuron.stop(neuron)
    end)

    :ok = Neuron.connect(left, %{1 => right})
    :ok = Neuron.connect(right, %{0 => left})

    assert %{
             index: 0,
             previous_activation: -1,
             activation: 1,
             local_field: 1.0,
             changed?: true
           } = Neuron.update(left)

    assert Neuron.activation(left) == 1
  end
end
