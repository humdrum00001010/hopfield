# Hopfield

이 프로젝트는 Energy Based Model을 이해하기 위한 공부 목적으로 만들어졌습니다.
Elixir에서 각 뉴런을 실제 프로세스로 만들고, 메시지 패싱으로 상태를 주고받는
전통적인 Hopfield Network를 구현합니다.

프로덕션용 신경망 라이브러리가 아니라, 다음 개념을 직접 보기 위한 작은 구현입니다.

* 기억은 어디에 저장되는가
* 주어진 cue에 대해 네트워크 전체가 어떻게 반응하는가
* recall은 어떻게 뉴런 상태를 바꾸는가
* energy는 recall과 어떤 관계가 있는가

## 표현

이 구현은 bipolar activation을 사용합니다.

```elixir
1   # 켜짐
-1  # 꺼짐 또는 반대 상태
```

하나의 memory, state, cue는 모두 같은 차원의 벡터입니다.

```elixir
[1, -1, 1, -1]
```

벡터의 길이가 곧 뉴런 개수입니다.

```text
memory length = state length = cue length = neuron count
```

## 기억 저장

Hopfield Network에서 정보는 개별 뉴런 안에 저장되지 않습니다.
정보는 뉴런 사이의 연결 강도인 weight matrix에 저장됩니다.

저장할 패턴을 memory라고 부릅니다.
memories로부터 Hebbian learning rule을 사용해 weight matrix를 만듭니다.

```text
w_ij = sum(memory_i * memory_j) / n
w_ii = 0
```

의미는 다음과 같습니다.

```text
두 뉴런이 보통 같은 부호를 가지면     -> positive weight
두 뉴런이 보통 반대 부호를 가지면     -> negative weight
자기 자신으로 가는 연결은 제거       -> w_ii = 0
```

즉, memory는 원본 벡터 그대로 저장되는 것이 아니라 뉴런 쌍 사이의 상관관계로 저장됩니다.

## Recall

Recall은 cue에서 시작해 뉴런 상태를 반복적으로 업데이트하면서 fixed point에 도달하는 과정입니다.

각 뉴런 `i`는 다른 뉴런들의 현재 상태를 받아 local field를 계산합니다.

```text
h_i = sum(w_ij * s_j)
s_i <- sign(h_i)
```

`h_i`가 양수이면 `s_i = 1`, 음수이면 `s_i = -1`이 됩니다.
`h_i = 0`이면 기존 상태를 유지합니다.

이 구현에서는 각 뉴런이 실제 Elixir 프로세스입니다.
뉴런은 peer 뉴런 프로세스에 현재 activation을 요청하고, 응답을 받은 뒤 자기 상태를 업데이트합니다.

## Energy

Hopfield Network는 Energy Based Model로 볼 수 있습니다.

현재 상태 `s`의 energy는 다음과 같이 계산합니다.

```text
E(s) = -1/2 * sum_i sum_j w_ij * s_i * s_j
```

낮은 energy는 현재 상태가 저장된 weight matrix와 잘 맞는다는 뜻입니다.
높은 energy는 현재 상태가 저장된 관계와 충돌한다는 뜻입니다.

비동기 업데이트에서는 energy가 증가하지 않습니다.
Recall은 cue에서 시작해서 energy가 낮아지는 방향으로 상태를 바꾸고,
더 이상 바뀌지 않는 fixed point에 도달하면 멈춥니다.

```text
cue = energy landscape 위의 시작점
weights = landscape의 모양
recall = downhill motion
memory = low-energy attractor
```

## 실행

```bash
mix test
iex -S mix
```

간단한 recall 예시:

```elixir
memory = [1, -1, 1, -1, 1]
cue = [1, -1, -1, -1, 1]

result = Hopfield.recall([memory], cue)

result.state
#=> [1, -1, 1, -1, 1]
```

단계별로 보고 싶다면:

```elixir
memories = [[1, -1, 1, -1]]
weight_matrix = Hopfield.hebbian_weights(memories)

network =
  Hopfield.Network.new(
    weight_matrix,
    [1, -1, -1, -1]
  )

Hopfield.Network.state(network)
Hopfield.Network.energy(network)
Hopfield.Network.asynchronous_sweep(network)
Hopfield.Network.state(network)
Hopfield.Network.energy(network)

Hopfield.Network.stop(network)
```

## 모듈

* `Hopfield` - Hebbian learning과 one-shot recall을 위한 public facade
* `Hopfield.Hebbian` - memories로부터 weight matrix를 만드는 Hebbian rule
* `Hopfield.Network` - weight matrix와 뉴런 프로세스 pid들을 들고 있는 plain struct
* `Hopfield.Neuron` - 하나의 뉴런을 나타내는 plain spawned process

뉴런 업데이트는 `Hopfield.Neuron.update/1`에 있습니다.
이 함수는 peer 뉴런들에게 activation을 요청하고, 응답을 모아 local field를 계산한 뒤
자신의 activation을 업데이트합니다.

사용하는 메시지는 다음과 같습니다.

```elixir
{:activation_request, requester, ref}
{:activation_reply, ref, index, activation}
```

## 참고 문헌

* Hopfield, J. J. (1982). "Neural networks and physical systems with emergent collective computational abilities." Proceedings of the National Academy of Sciences, 79(8), 2554-2558. https://doi.org/10.1073/pnas.79.8.2554
* Little, W. A. (1974). "The existence of persistent states in the brain." Mathematical Biosciences, 19(1-2), 101-120. https://doi.org/10.1016/0025-5564(74)90031-5
* Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). "Spin-glass models of neural networks." Physical Review A, 32(2), 1007-1018. https://doi.org/10.1103/PhysRevA.32.1007
* Hebb, D. O. (1949). The Organization of Behavior: A Neuropsychological Theory. Wiley. 학습 규칙의 역사적 배경입니다. 논문이 아니라 책입니다.
