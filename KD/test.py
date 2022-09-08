import torch
from KD.utils import teacher_selector, output_selector
import sys
from KD.loss import betweenLoss, CrossEntropy

def test(teachers, best_model,test_loader):
    best_model.eval()
    test_loss = 0
    teacher_correct = 0
    correct = 0
    total = 0
    gamma = '[1,1,1,1,1]'
    criterion = betweenLoss(eval(gamma), loss = CrossEntropy)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            total += targets.size(0)
            # Get output from student model
            feature_student, outputs = best_model(inputs)
            # Get teacher model
            teacher = teacher_selector(teachers, 0)
            # Get output from teacher model
            feature_teacher, answers = teacher(inputs)
            # Select output from student and teacher
            outputs, answers = output_selector([feature_student, outputs], [feature_teacher, answers], [0, 1])
            # Calculate loss between student and teacher
            loss = criterion(outputs, answers)
            # Calculate loss for discriminators
            test_loss += loss.item()
            _, predicted = outputs[-1].max(1)
            _, teacher_predicted = answers[-1].max(1)
            teacher_correct += teacher_predicted.eq(targets).sum().item()
            correct += predicted.eq(targets).sum().item()
            sys.stdout.write('\r[Test] [iter: %d / all %d], Teacher: %s, G_Loss: %.3f, Student Acc: %.3f%% (%d/%d), Teacher Acc: %.3f%% (%d/%d)' \
                    % (batch_idx + 1, len(test_loader), 'Transformer', test_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                    100. * teacher_correct / total, teacher_correct, total))
            sys.stdout.flush()  
        acc = 100. * correct / total
    return acc