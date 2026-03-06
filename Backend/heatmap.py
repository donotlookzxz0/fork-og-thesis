import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from services.ai_recommendations import state
from app import app

with app.app_context():
    if state.model is None or not state.user_map or not state.item_map:
        print("Model not trained yet")
        exit()
    
    n_users = len(state.user_map)
    n_items = len(state.item_map)
    
    with torch.no_grad():
        user_emb = state.model.user_emb.weight
        item_emb = state.model.item_emb.weight
        score_matrix = torch.mm(user_emb, item_emb.t()).numpy()
    
    inv_user_map = {idx: uid for uid, idx in state.user_map.items()}
    inv_item_map = {idx: iid for iid, idx in state.item_map.items()}
    
    max_display = 20
    n_users_display = min(max_display, n_users)
    n_items_display = min(max_display, n_items)
    
    user_labels = [f"U{inv_user_map[i]}" for i in range(n_users_display)]
    item_labels = [f"I{inv_item_map[i]}" for i in range(n_items_display)]
    
    display_matrix = score_matrix[:n_users_display, :n_items_display]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        display_matrix,
        xticklabels=item_labels,
        yticklabels=user_labels,
        cmap='RdYlGn',
        center=0,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Dot Product Score'}
    )
    plt.title('User-Item Recommendation Scores Heatmap')
    plt.xlabel('Items')
    plt.ylabel('Users')
    plt.tight_layout()
    
    plt.savefig('recommendation_heatmap.png', dpi=150)
    print(f"Heatmap saved to recommendation_heatmap.png")
    print(f"Total users: {n_users}, Total items: {n_items}")
    print(f"Showing: {n_users_display} users x {n_items_display} items")
    
    plt.show()