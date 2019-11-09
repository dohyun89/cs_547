import torch
from utils import*
from discriminator import *
from generator import *
from torch.autograd import Variable
import time
import numpy as np

root = './cifar10'
batch_size = 128
num_workers = 8
num_epoch = 200
gen_train = 1
n_z = 100
n_classes = 10
load_models = True

trainloader, testloader = dataloader(root = root, 
                                    train_batch_size= batch_size,
                                    test_batch_size = batch_size,  
                                    num_workers = num_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if load_models == True:
    aD = torch.load(tempD.model)
    aD = aD.to(device)

    aG = torch.load(tempG.model)
    aG = aD.to(device)

else:
    aD = discriminator()
    aD = aD.to(device)

    aG = generator()
    aG = aG.to(device)

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))
criterion = nn.CrossEntropyLoss()

#Noise
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).to(device)

loss1 = []
loss2 = []
loss3 = []
loss4 = []
loss5 = []
acc1 = []
start_time = time.time()
# Train the model
for epoch in range(16,num_epoch):
    for group in optimizer_d.param_groups:
        for p in group['params']:
            state = optimizer_d.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000

    for group in optimizer_g.param_groups:
        for p in group['params']:
            state = optimizer_g.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000

    aG.train()
    aD.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        if((batch_idx%gen_train)==0):
            for p in aD.parameters():
                p.requires_grad_(False)

            aG.zero_grad()

            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()

            fake_data = aG(noise)
            gen_source, gen_class  = aD(fake_data)

            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            gen_cost.backward()

            optimizer_g.step()
        
        # train D
        for p in aD.parameters():
            p.requires_grad_(True)

        aD.zero_grad()

        # train discriminator with input from generator
        label = np.random.randint(0,n_classes,batch_size)
        noise = np.random.normal(0,1,(batch_size,n_z))
        label_onehot = np.zeros((batch_size,n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise)

        disc_fake_source, disc_fake_class = aD(fake_data)

        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)

        # train discriminator with input from the discriminator
        real_data = Variable(X_train_batch).cuda()
        real_label = Variable(Y_train_batch).cuda()

        disc_real_source, disc_real_class = aD(real_data)

        prediction = disc_real_class.data.max(1)[1]
        accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0

        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)

        gradient_penalty = calc_gradient_penalty(batch_size, aD,real_data,fake_data)

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()

        optimizer_d.step()
    
    loss1.append(gradient_penalty.item())
    loss2.append(disc_fake_source.item())
    loss3.append(disc_real_source.item())
    loss4.append(disc_real_class.item())
    loss5.append(disc_fake_class.item())
    acc1.append(accuracy)
    if((batch_idx%50)==0):
        print("Training\n Epoch:",epoch, "Batch:",batch_idx, "%.2f" % np.mean(loss1), 
                                "%.2f" % np.mean(loss2), 
                                "%.2f" % np.mean(loss3), 
                                "%.2f" % np.mean(loss4), 
                                "%.2f" % np.mean(loss5), 
                                "%.2f" % np.mean(acc1))
    
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('Testing',accuracy_test, '| Time: ',time.time()-start_time)

    ### save output
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()

    fig = plot(samples)
    plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)

    if(((epoch+1)%1)==0):
        torch.save(aG,'tempG.model')
        torch.save(aD,'tempD.model')