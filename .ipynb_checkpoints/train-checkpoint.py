import pandas as pd
import tez
from sklearn import model_selection
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing
import numpy as np


class RecipeDataset():
    def __init__(self, users, recipes, ratings):
        self.users = users
        self.recipes = recipes
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        user = self.users[item]
        recipe = self.recipes[item]
        rating = self.ratings[item]

        return{"users": torch.tensor(user, dtype=torch.long),
               "recipes": torch.tensor(recipe, dtype=torch.long),
               "ratings": torch.tensor(rating, dtype=torch.float)
               }


class RecSysModel(tez.Model):
    def __init__(self, num_users, num_recipes):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, 32)
        self.recipes_embed = nn.Embedding(num_recipes, 32)
        self.out = nn.Linear(64, 1)
        self.step_scheduler_after = "epoch"

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.7)
        return sch

    def monitor_metrics(self, output, rating):
        output = output.detach().cpu().numpy()
        rating = rating.detach().cpu().numpy()
        return {
            'rmse': np.sqrt(metrics.mean_squared_error(rating, output))
        }

    def forward(self, users, recipes, ratings):
        user_embeds = self.user_embed(users)
        recipe_embeds = self.recipe_embed(recipes)
        output = torch.cat([user_embeds, recipe_embeds], dim=1)
        output = self.out(output)
        metrics = {}
        loss = nn.MSELoss()(output, ratings.view(-1, 1))
        metrics = self.monitor_metrics(output, ratings.view(-1, 1))
        return output, loss, metrics


def train():
    df = pd.read_parquet("food.parquet")
    # All Food Columns from Name, review count, review descriptions, etc
    lbl_user = preprocessing.LabelEncoder()
    lbl_recipe = preprocessing.LabelEncoder()

    df.user = lbl_user.fit_transform(df.user.values)
    df.recipe = lbl_recipe.fit_transform(df.recipe.values)

    df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.Rating.values
                                                          )

    train_dataset = RecipeDataset(
        users=df_train.user.values, recipes=df_train.recipe.values, ratings=df_train.rating.values)
    valid_dataset = RecipeDataset(
        users=df_valid.user.values, recipes=df_valid.recipe.values, ratings=df_valid.rating.values)

    model = RecSysModel(num_users=len(lbl_user.classes_),
                        num_recipes=len(lbl_recipe.classes_))
    model.fit(
        train_dataset,
        valid_dataset,
        train_bs=1024,
        valid_bs=1024,
        fp16=True
    )


if __name__ == "__main__":
    train()
