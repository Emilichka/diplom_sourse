import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

# interpolate_input

def loss_func(output, y_batch):
    return torch.nn.NLLLoss()(torch.transpose(output, 2, 1), y_batch)

def interpolate_path(baseline,
                       region,
                       alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    baseline_x = baseline
    input_x = region
    delta = input_x - baseline_x
    path = baseline_x +  alphas_x * delta
    for elem  in path[1:]:
          elem[:, :4]=region[0][:, :4]
    return path

def calculate_outputs_and_gradients(inputs_path, model,target):
    gradients = []
    predictions = []
    for reg in inputs_path:
        with torch.autograd.set_grad_enabled(True):
            reg = torch.unsqueeze(reg, 0)
            region = Variable(reg.data, requires_grad=True)
            output = model(region.cuda())
            pred = torch.argmax(output, dim=2)
            model.zero_grad()
            loss = loss_func(output, target.long().cuda())
            loss.backward()
            grad_i = region.grad.detach().cpu().numpy()[0]
            gradients.append(grad_i)
            predictions.append(int(pred.any()))
    return gradients, predictions


def good_plot(prob, step, region, i):
    alphas = torch.linspace(start=0.0, end=1.0, steps=step+1)
    plt.plot(alphas, prob)
    plt.title('Target class predicted probability over alpha')
    plt.ylabel('model p(target class)')
    plt.xlabel('alpha')
    plt.ylim([0, 1])
    plt.savefig(f'prediction_changes/prediction_for_{i}_region.png')
    
   
def integrated_gradients(region, model, target,steps=50):
    m_step=50
    baseline = torch.zeros(size=region.shape)
    alphas = torch.linspace(start=0.0, end=1.0, steps=m_step+1)
    path = interpolate_images(baseline, region, alphas)
    grad, pred = calculate_outputs_and_gradients(path, model, target)
    avg_grads = np.average(grad[:-1], axis=0)
    delta = (region - baseline).squeeze(0).cpu().detach().numpy()
    integrated_grad = delta * avg_grads
    return integrated_grad, pred