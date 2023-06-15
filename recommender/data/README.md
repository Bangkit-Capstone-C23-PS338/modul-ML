Data source: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

Data synthesized to match with requirements:
* Instagram followers synthesized from members * 50, with Youtube and Tiktok followers adjusted to be around Instagram followers
* Some synthesized data retain categories from the real data, and some data has aggregated categories using correlation score (correlation can be seen in "others" folder, while aggregation process using Hierarchical Clustering can be seen in "../misc/clustered_category.py"). Clustering result can be seen in "others/clustered.csv", with some adjusted result.
* Pricing is randomized, with those with high average followers has higher chance to have HIGH pricing + lower chance to have LOW pricing and vice-versa