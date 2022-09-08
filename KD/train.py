from torch.autograd import Variable
import torch
from KD.utils import teacher_selector, output_selector
import sys
import random
from KD.loss import betweenLoss, discriminatorLoss, CrossEntropy

def train(teachers, student, discriminators, epoch, target_loader, valid_loader, optimizer, scheduler, lambda_factor):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    student.train(True)
    gamma = '[1,1,1,1,1]'
    eta = '[1,1,1,1,1]'
    train_loss = 0
    valid_loss = 0
    correct = 0
    total = 0
    discriminator_loss = 0
    criterion = betweenLoss(eval(gamma), loss= CrossEntropy)
    discriminators_criterion = discriminatorLoss(discriminators, eval(eta))

    target = list(enumerate(target_loader))
    train_steps = len(target)

    for batch_idx in range(train_steps):
        _, (target_data, target_label) = target[batch_idx] # unsupervised learning
        total += target_data.shape[0]
        target_data, target_label = Variable(target_data), Variable(target_label)

        optimizer.zero_grad()
        # Get output from student model
        feature_student, outputs = student(target_data)
        # Get teacher model
        teacher = teacher_selector(teachers, random.choices([0, 1], weights=(50, 50), k =1)[0])
        # Get output from teacher model
        feature_teacher, answers = teacher(target_data)
        # Select output from student and teacher
        otputs, answers = output_selector([feature_student, outputs], [feature_teacher, answers], [0,1])
        # Calculate loss between student and teacher
        loss = criterion(outputs, answers)
        # Calculate loss for discriminators
        d_loss = discriminators_criterion(outputs, answers)
        # Get total loss
        total_loss = loss + lambda_factor * d_loss
        # total_loss = loss
        total_loss.backward()
        optimizer.step()

        train_loss += loss.item()
        discriminator_loss += d_loss.item()
        _, predicted = outputs[-1].max(1)
        correct += predicted.eq(target_label).sum().item()

        train_acc = 100. * correct / total

        sys.stdout.write('\r[Train] [iter: %d / all %d], Teacher: %s, Lr: %.3f, G_Loss: %.3f, D_Loss: %.3f, Acc: %.3f%% (%d/%d)' \
              % (batch_idx + 1, len(target_loader), 'Transformer', scheduler.get_lr()[0], train_loss / (batch_idx + 1), discriminator_loss / (batch_idx + 1), train_acc, correct, total))
        sys.stdout.flush()

    print()
    total = 0
    correct = 0
    teacher_correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            total += targets.size(0)
            # Get output from student model
            feature_student, outputs = student(inputs)
            # Get teacher model
            teacher = teacher_selector(teachers, 0)
            # Get output from teacher model
            feature_teacher, answers = teacher(inputs)
            # Select output from student and teacher
            outputs, answers = output_selector([feature_student, outputs], [feature_teacher, answers], [0, 1])
            # Calculate loss between student and teacher
            loss = criterion(outputs, answers)

            valid_loss += loss.item()
            _, predicted = outputs[-1].max(1)
            _, teacher_predicted = answers[-1].max(1)

            teacher_correct += teacher_predicted.eq(targets).sum().item()
            correct += predicted.eq(targets).sum().item()
            sys.stdout.write('\r[Valid] [iter: %d / all %d], Teacher: %s, G_Loss: %.3f, Student Acc: %.3f%% (%d/%d), Teacher Acc: %.3f%% (%d/%d)' \
                % (batch_idx + 1, len(valid_loader), 'Transformer', valid_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                100. * teacher_correct / total, teacher_correct, total))
            sys.stdout.flush()  
    val_acc = 100. * correct / total
    return val_acc