import matplotlib.pyplot as plt
import numpy as np

# =========================
# FINAL EVALUATION RESULTS
# =========================

episodes = 50
goals = 36
crashes = 14

success_rate = goals / episodes * 100

print("Episodes:", episodes)
print("Goals:", goals)
print("Crashes:", crashes)
print("Success Rate:", round(success_rate,2), "%")


# =========================
# BAR CHART
# =========================

plt.figure(figsize=(6,4))

plt.bar(["Goals","Crashes"], [goals,crashes])

plt.title("Navigation Results")
plt.ylabel("Count")

plt.savefig("plots/goals_vs_crashes.png")

plt.close()


# =========================
# PIE CHART
# =========================

plt.figure(figsize=(6,6))

plt.pie(
    [goals, crashes],
    labels=["Goals","Crashes"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Navigation Success Rate")

plt.savefig("plots/success_rate.png")

plt.close()


# =========================
# EPISODE OUTCOME GRAPH
# =========================

# 1 = goal, 0 = crash
results = [1]*goals + [0]*crashes

np.random.shuffle(results)

plt.figure(figsize=(10,4))

plt.plot(results, marker="o")

plt.title("Episode Outcomes")
plt.xlabel("Episode")
plt.ylabel("Result (1=Goal, 0=Crash)")

plt.savefig("plots/episode_outcomes.png")

plt.close()

print("Plots saved in /plots folder")