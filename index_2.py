# Load from helpers
from helpers.drawPlotStatistics import plot_training_history

# separated functions
from packages.buildCourses import load_courses
from packages.buildUsers import load_users
from controllers.processData import preprocess_data, generate_category_embeddings
from controllers.mappingData import generate_sync_data
from controllers.buildingData import build_foundation_model
from controllers.recommendation import recommend_courses, recommendation_display
from models.baseModel import train_model, retrain_model

async def validateData():
    # Load and validate data
        courses = await load_courses("./data/course.json")
        users = await load_users("./data/user.json")

        # Ensure that the loaded data is valid
        if not courses or not users:
            print("No valid courses or users data found.")
            exit(0)
            return
        
        return {
            'courses': courses,
            'users': users
        }

# Bootstrap the process
async def main():
    try:
        # Load And validate data
        loadedData = await validateData()

        # Preprocess data structure
        data = preprocess_data(loadedData['users'], loadedData['courses'])
        user_vectors, course_vectors = data['userVectors'], data['courseVectors']

        # Vectorize course data with TF-IDF
        course_vectors = generate_category_embeddings(course_vectors)

        # generate matrix and vectorize all the data
        mock_data = generate_sync_data(user_vectors, course_vectors)
        xs_users, xs_courses, ys = mock_data['xsUsers'], mock_data['xsCourses'], mock_data['ys']

        # Train the model
        model = build_foundation_model(user_vectors, course_vectors, 1e-3, 16, 0.01, 0.2)
        results = await train_model(model, xs_users, xs_courses, ys)

        # Show training chart result
        plot_training_history(results)

        # Retrain the model after saving
        resultReloaded = await retrain_model(xs_users, xs_courses, ys, 1e-4)

        # Show re-training chart result
        plot_training_history(resultReloaded)

        # Example recommendation for first user
        recommendations = recommend_courses(user_vectors[4]['vector'], course_vectors)
        # display the recommendation result
        recommendation_display(recommendations)

    except Exception as e:
        print(f"Error in main: {e}")

# Run the script
import asyncio
asyncio.run(main())
