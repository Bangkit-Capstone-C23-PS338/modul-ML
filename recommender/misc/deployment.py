# Try deployment
import pandas as pd
import numpy as np
import multiprocessing

STAR_WEIGHT = 0.5
SENTIMENT_WEIGHT = 0.5
STARTING_PROFILE_SCORE = 0.5
COLD_START_AVG_REVIEW = 0.5

inf_data = [
    {
        'id': 100,
        'categories': ["Category 1", "Category 2", "Category 3"],
        'instagram': 10000000,
        'youtube': 10000000,
        'tiktok': 10000000,
        'product': [
            {
                'id': 1,
                'price': 1000000000
            },
            {
                'id': 2,
                'price': 1000000000
            },
            {
                'id': 3,
                'price': 5000000000
            },
        ]
    },
    {
        'id': 200,
        'categories': ["Category 5"],
        'instagram': 10,
        'tiktok': 10,
        'product': [
            {
                'id': 1,
                'price': 100
            },
            {
                'id': 2,
                'price': 100
            }
        ]
    },
    {
        'id': 300,
        'categories': ["Category 1", "Category 2", "Category 3", "Category 4"],
        'instagram': 10000000,
        'youtube': 10000000,
        'tiktok': 10000000,
        'product': [
            {
                'id': 1,
                'price': 100000000
            },
            {
                'id': 2,
                'price': 10000000000
            },
            {
                'id': 3,
                'price': 5000000000000
            },
        ]
    },
    {
        'id': 400,
        'categories': ["Category 1", "Category 2", "Category 3", "Category 4"],
        'instagram': 10000,
        'youtube': 10000,
        'tiktok': 10000,
        'product': [
            {
                'id': 1,
                'price': 1000000
            },
            {
                'id': 2,
                'price': 10000000
            },
            {
                'id': 3,
                'price': 5000
            },
        ]
    }
]

own_data = [
    {
        'id': 1,
        'categories': ["Category 1", "Category 2", "Category 3", "Category 4"]
    },
    {
        'id': 2,
        'categories': ['Category 6', 'Category 10']
    },
    {
        'id': 3,
        'categories': ['Category 3']
    },
]

reviews = [
    {
        'own_id': 1,
        'inf_id': 100,
        'rating': 5,
        'review': "Kontennya bagus! Professional!",
        'sentiment_rating': 1
    },
    {
        'own_id': 1,
        'inf_id': 300,
        'rating': 5,
        'review': "Lumayan, responnya cepet",
        'sentiment_rating': 1
    },
    # {
    #     'own_id': 2,
    #     'inf_id': 200,
    #     'rating': 5,
    #     'review': "OK!",
    #     'sentiment_rating': 0.5
    # },
]

INF_PROFILE = ['avg_rating', 'pricing_LOW', 'pricing_BELOW_AVG', 'pricing_AVG',
       'pricing_ABOVE_AVG', 'pricing_HIGH', 'Category 1', 'Category 10',
       'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6',
       'Category 7', 'Category 8', 'Category 9', 'youtube_High', 'youtube_Low',
       'youtube_Medium', 'tiktok_High', 'tiktok_Low', 'tiktok_Medium',
       'insta_follower_High', 'insta_follower_Low', 'insta_follower_Medium']

USER_PROFILE = ['pricing_LOW', 'pricing_BELOW_AVG', 'pricing_AVG', 'pricing_ABOVE_AVG',
       'pricing_HIGH', 'Category 1', 'Category 10', 'Category 2', 'Category 3',
       'Category 4', 'Category 5', 'Category 6', 'Category 7', 'Category 8',
       'Category 9', 'youtube_High', 'youtube_Low', 'youtube_Medium',
       'tiktok_High', 'tiktok_Low', 'tiktok_Medium', 'insta_follower_High',
       'insta_follower_Low', 'insta_follower_Medium']


# < 1jt	LOW
# 1jt - 5jt	BELOW_AVERAGE
# 5jt - 10jt	AVERAGE
# 10jt - 20jt	ABOVE_AVERAGE
# >20jt	HIGH

# PR
def get_review_from_own_id(own_id):
    result = []
    for review in reviews:
        if (review['own_id'] == own_id):
            result.append(review)

    return result

# get_business_owner() -> get categorynya
def get_categories_from_own_id(own_id):
    for data in own_data:
        if (data['id'] == own_id):
            return data['categories']
        
# 1. get_influencer_by_username() -> {'avg_rating': 4.5} / 5
# 2. get_influencer_review() -> looping
def get_average_rating(inf_id):
    result = 0
    i = 0
    for review in reviews:
        if (review['inf_id'] == inf_id):
            result += STAR_WEIGHT * review['rating'] / 5+ SENTIMENT_WEIGHT * review['sentiment_rating']
            i += 1

    if (i != 0):
        return result / i
    else:
        return COLD_START_AVG_REVIEW

def get_influencer_recommender_profile(inf_id):
    YOUTUBE_HIGH_THRES = 2_000_000
    YOUTUBE_LOW_THRES = 100_000
    TIKTOK_HIGH_THRES = 2_000_000
    TIKTOK_LOW_THRES = 100_000
    INSTAGRAM_HIGH_THRES = 1_000_000
    INSTAGRAM_LOW_THRES = 50_000

    influencer = None
    for inf in inf_data:
        if (inf['id'] == inf_id):
            influencer = inf
            break

    if (influencer == None):
        print("Influencer not found")
        return
    
    inf_profile = pd.DataFrame(STARTING_PROFILE_SCORE, index=[inf_id], columns=INF_PROFILE, dtype=float)
    inf_profile.loc[inf_id][influencer['categories']] = 1

    # One hot followers
    if (influencer.get('youtube', 0) > YOUTUBE_HIGH_THRES):
        inf_profile.loc[inf_id]['youtube_High'] = 1
    elif (influencer.get('youtube', 0) > YOUTUBE_LOW_THRES):
        inf_profile.loc[inf_id]['youtube_Medium'] = 1
    else:
        inf_profile.loc[inf_id]['youtube_Low'] = 1

    if (influencer.get('instagram', 0) > INSTAGRAM_HIGH_THRES):
        inf_profile.loc[inf_id]['instagram_High'] = 1
    elif (influencer.get('instagram', 0) > INSTAGRAM_LOW_THRES):
        inf_profile.loc[inf_id]['instagram_Medium'] = 1
    else:
        inf_profile.loc[inf_id]['instagram_Low'] = 1
    
    if (influencer.get('tiktok', 0) > TIKTOK_HIGH_THRES):
        inf_profile.loc[inf_id]['tiktok_High'] = 1
    elif (influencer.get('tiktok', 0) > TIKTOK_LOW_THRES):
        inf_profile.loc[inf_id]['tiktok_Medium'] = 1
    else:
        inf_profile.loc[inf_id]['tiktok_Low'] = 1

    # Get price categories
    for product in influencer['product']:
        if (product['price'] > 20_000_000):
            inf_profile.loc[inf_id]['price_HIGH'] = 1
        elif (product['price'] > 10_000_000):
            inf_profile.loc[inf_id]['price_ABOVE_AVG'] = 1
        elif (product['price'] > 5_000_000):
            inf_profile.loc[inf_id]['price_AVG'] = 1
        elif (product['price'] > 1_000_000):
            inf_profile.loc[inf_id]['price_BELOW_AVG'] = 1
        else:
            inf_profile.loc[inf_id]['price_LOW'] = 1

    inf_profile['avg_rating'] = get_average_rating(inf_id)

    return inf_profile
        
def get_all_influencer_recommender_profile():
    inf_profiles = []
    for influencer in inf_data:
        inf_profiles.append(get_influencer_recommender_profile(influencer['id'])) 
    return inf_profiles
    

def get_combined_rating(review):
    return STAR_WEIGHT * review['rating'] / 5 + SENTIMENT_WEIGHT * review['sentiment_rating']

def get_user_recommender_profile(own_id):
    user_profile = pd.DataFrame(STARTING_PROFILE_SCORE, index=[own_id], columns=USER_PROFILE, dtype=float)
    categories = get_categories_from_own_id(own_id)

    if (categories == None):
        print("Invalid owner ID")
        return

    user_profile.loc[own_id][categories] = 1

    user_reviews = get_review_from_own_id(own_id)

    # print(user_reviews)
    for review in user_reviews:
        user_profile = user_profile.add((get_influencer_recommender_profile(review['inf_id']).drop("avg_rating", axis=1) * get_combined_rating(review)).values)

    user_profile = user_profile / (len(user_reviews)+1)

    return user_profile
 

# Inference according to own_id
def get_influencer_scores(own_id):
    import tensorflow as tf

    MODEL = "recommender-smote-try-simple-I"

    export_path = f"recommender/log/model/savedmodel/{MODEL}/"
    model = tf.saved_model.load(export_path)
    infer = model.signatures["serving_default"]

    inf_profile = get_all_influencer_recommender_profile()
    user_profile = get_user_recommender_profile(own_id)

    id = [inf.index for inf in inf_profile]
    inputs = [{'inf_feature': tf.convert_to_tensor(inf, dtype=float), 
              'own_feature': tf.convert_to_tensor(user_profile, dtype=float)}
              for inf in inf_profile]
    
    score = []
    for i, data in enumerate(inputs):
        score.append(infer(**data))
        print(f"UserID: {own_id}, Inf ID: {id[i].values[0]} ->", infer(**data)['dot_2'].numpy()[0, 0])


get_influencer_scores(3)
# correlation
# {
#     "id": 1,
#     "influencers": ['influencer1', 'influencer2', Influencer(300)]
# }
