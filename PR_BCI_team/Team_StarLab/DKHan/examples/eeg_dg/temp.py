lr = 1
warm_up_step = 10
lambda1 = lambda epoch: epoch*(lr/warm_up_step)

for epoch in range(0,warm_up_step):

    print(lambda1(epoch))