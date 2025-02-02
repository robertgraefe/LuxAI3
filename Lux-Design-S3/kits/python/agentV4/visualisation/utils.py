def plot_training(reward_history):
    # Calculate the Simple Moving Average (SMA) with a window size of 50
    sma = np.convolve(reward_history, np.ones(50) / 50, mode='valid')

    # Clip max (high) values for better plot analysis
    _reward_history = np.clip(reward_history, a_min=None, a_max=100)
    sma = np.clip(sma, a_min=None, a_max=100)

    plt.figure(1)
    plt.clf()
    plt.title("Obtained Rewards")
    plt.plot(_reward_history, label='Raw Reward', color='#4BA754', alpha=1)
    plt.plot(sma, label='SMA 50', color='#F08100')
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()

    plt.tight_layout()
    plt.grid(True)
    plt.pause(0.001)

    save_chart("Reward", "Roll Reward", "Step", _reward_history.tolist(), sma.tolist(), [i for i in range(_reward_history.__len__())])

def save_chart(x1_label: str, x2_label: str, y_label: str, x1: list, x2: list, y: list):
    with open('/home/robert/PycharmProjects/LuxAI3/GUI/src/assets/data.json', 'w', encoding='utf-8') as file:
        dump = {
            "x1_label": x1_label,
            "x2_label": x2_label,
            "y_label": y_label,
            "x1": x1,
            "x2": x2,
            "y": y
        }
        json.dump(dump, file)