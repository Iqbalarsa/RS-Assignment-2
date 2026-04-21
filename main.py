import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import SASRec
from utils import WarpSampler, evaluate, evaluate_valid, preprocess_ml1m, split_data
import numpy as np

class Args:
    dataset = 'ratings.dat' # Path
    batch_size = 128
    lr = 0.001
    maxlen = 100          # Maximal Length of Sequence
    hidden_units = 50     
    num_blocks = 2        # Number of blocks in transformers
    num_heads = 1
    dropout_rate = 0.2
    l2_emb = 0.0          # Regularisation
    num_epochs = 100     
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    norm_first = False
    patience = 5

args = Args()

if __name__ == '__main__':
    # Loading and Splitting Data 
    user_seqs = preprocess_ml1m(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum = split_data(user_seqs)
    
    dataset = [user_train, user_valid, user_test, usernum, itemnum]

    print(f"Data Loaded: User: {usernum}, Item: {itemnum}")

    # Setup Sampler (for automatic negative sampling)
    sampler = WarpSampler(user_train, usernum, itemnum, 
                          batch_size=args.batch_size, 
                          maxlen=args.maxlen, 
                          n_workers=3)

    # Call the Model
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    # Initialize the weight 
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    # Loss and Optimizer
    # BCE
    bce_criterion = nn.BCEWithLogitsLoss()
    adam_optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # Training Loop
    num_batch = len(user_train) // args.batch_size
    
    best_ndcg = 0.0
    early_stop_counter = 0
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0
        
        for step in range(num_batch):
            # Taking btach from sampler
            u, seq, pos, neg = sampler.next_batch()
            
            # Convert to tensor
            seq = torch.as_tensor(np.array(seq), dtype=torch.long, device=args.device)
            pos = torch.as_tensor(np.array(pos), dtype=torch.long, device=args.device)
            neg = torch.as_tensor(np.array(neg), dtype=torch.long, device=args.device)
            
            pos_logits, neg_logits = model(u, seq, pos, neg)
            
            # Masking for padding
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            
            indices = (pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            # Regularization
            if args.l2_emb > 0:
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)

            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            
            epoch_loss += loss.item()

        # EValuate
        model.eval()
        with torch.no_grad(): 
            t_valid = evaluate_valid(model, dataset, args)
            t_test = evaluate(model, dataset, args)
            
        current_ndcg = t_valid[0] # NDCG@10

        print(f"Epoch {epoch} | Valid NDCG@10: {current_ndcg:.4f}")

        # Early Stopping
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            early_stop_counter = 0 # Reset counter 
            # Save the best model
            torch.save(model.state_dict(), 'best_sasrec.pth')
            print("There is an improvement, save checkpoint...")
        else:
            early_stop_counter += 1
            print(f"There is no improvement ({early_stop_counter}/{args.patience})")

        # Valid data
        print(f"Valid: NDCG@10: {t_valid[0]:.4f}, Recall@10: {t_valid[1]:.4f}, NDCG@20: {t_valid[2]:.4f}, Recall@20: {t_valid[3]:.4f}")
        # Test data
        print(f"Test : NDCG@10: {t_test[0]:.4f}, Recall@10: {t_test[1]:.4f}, NDCG@20: {t_test[2]:.4f}, Recall@20: {t_test[3]:.4f}")


        if early_stop_counter >= args.patience:
            print(f"Early stopping was triggered by epoch {epoch}. Training has stopped.")
            break 
        
       
    sampler.close()
    print("Training has completed!")
