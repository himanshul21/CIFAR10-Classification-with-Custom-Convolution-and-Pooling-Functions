import torch

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        # Compute the softmax of the logits
        # Subtracting max for numerical stability
        print("logits : ", logits)
        print("targets : ", targets)
        exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])
        softmax_probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
        
        # Compute the negative log likelihood of the true class
        # print(softmax_probs[range(logits.size(0)), targets])
        neg_log_likelihood = -torch.log(softmax_probs[range(logits.size(0)), targets])
        
        # Compute the mean loss
        loss = torch.mean(neg_log_likelihood)
        
        return loss