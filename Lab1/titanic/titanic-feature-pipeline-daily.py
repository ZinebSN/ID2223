
import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_person(survival,age_min, age_max):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"pclass": [random.choice([1,2,3])],
                       "sex": [random.choice([1,2])],
                       "age": [random.uniform(age_max, age_min)],
                       "parch": [random.choice([0,1,2])],
                       "embarked": [random.choice([0,1,2])]
                      })
    df['survived'] = survival
    return df


def get_random_person():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    surv_df = generate_person(1, 15, 40)
    dead_df = generate_person(0, 50,70)
    
    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.uniform(0,2)
    #pick_random=1
    if pick_random >= 2:
        per_df = surv_df
        print("survived added")
    else:
        per_df = dead_df
        print("dead added")
    
    return per_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login() 
    fs = project.get_feature_store()

    titanic_df = get_random_person()

    titanic_fg = fs.get_feature_group(name="titanic_modal",version=1)
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()