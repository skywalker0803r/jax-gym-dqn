# JAX Gymnasium DQN

依照 `JAX_Gymnasium_RL_Development_Plan.pdf` 實作的 CartPole-v1 DQN，使用 JAX、Flax、Optax 與 Gymnasium。

## 安裝

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 使用 CUDA 12 JAX wheel，請在 WSL2/Linux 環境安裝。原生 Windows 的 JAX pip wheel 目前只會提供 CPU backend。

在 Windows 先安裝 WSL2：

```powershell
wsl --install -d Ubuntu
```

進入 Ubuntu 後安裝並確認 CUDA：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

## 執行

```powershell
python -m unittest discover -s tests -v
python -m jax_rl_cartpole.train --episodes 200 --require-cuda
python -m jax_rl_cartpole.evaluate --episodes 10
```

訓練會將參數保存至 `checkpoints/dqn_params.pkl`。程式預設設定 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.60`，並將訓練更新與 loss 計算封裝在 JIT 純函數中。