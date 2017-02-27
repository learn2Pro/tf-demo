# 训练集
# 每个样本点有3个分量 (x0,x1,x2)
x = [(1, 0., 3), (1, 1., 3), (1, 2., 3), (1, 3., 2), (1, 4., 4)]
# y[i] 样本点对应的输出
y = [95.364, 97.217205, 75.195834, 60.105519, 49.342380]

# 迭代阀值，当两次迭代损失函数之差小于该阀值时停止迭代
epsilon = 0.0001

# 学习率
alpha = 0.01
diff = 0
max_itor = 1000
error1 = 0
error0 = 0
cnt = 0
m = len(x)

t0 = 0
t1 = 0
t2 = 0

while True:

    for i in range(m):
        diff = (t0 * x[i][0] + t1 * x[i][1] + t2 * x[i][2]) - y[i]
        t0 -= diff * alpha * x[i][0]
        t1 -= diff * alpha * x[i][1]
        t2 -= diff * alpha * x[i][2]

    error1 = 0
    for i in range(len(x)):
        error1 += ( y[i] - (t0 * x[i][0] + t1 * x[i][1] + t2 * x[i][2]))**2/2

    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1
    print(' theta0 : %f, theta1 : %f, theta2 : %f, error1 : %f' % (t0, t1, t2, error1))
