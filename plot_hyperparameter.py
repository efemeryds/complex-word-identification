import matplotlib.pyplot as plt

learning_rate = [0.00005, 0.0005, 0.005, 0.05, 0.07, 0.09, 0.5, 1]
f1_weighted_average = [0.75, 0.84, 0.86, 0.87, 0.86, 0.86, 0.77, 0.75]


plt.plot(learning_rate, f1_weighted_average)
plt.xscale('log')
plt.title('Learning rate vs F1 score')
plt.xlabel('Learning rate')
plt.ylabel('F1 weighted average')
plt.savefig("learning_rate.png")
plt.show()






