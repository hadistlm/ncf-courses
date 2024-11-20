# Load from helpers
from helpers.drawPlotStatistics import plot_training_history

# separated functions
from packages.buildCourses import load_courses
from packages.buildUsers import load_users
from controllers.processData import preprocess_data
from controllers.mappingData import generate_sync_data
from controllers.buildingData import build_foundation_model
from controllers.recommendation import recommend_courses
from models.baseModel import train_model, retrain_model

# Bootstrap the process
async def main():
    try:
        # Load and validate data
        courses = await load_courses("./data/course.json")
        users = await load_users("./data/user.json")

        # Ensure that the loaded data is valid
        if not courses or not users:
            print("No valid courses or users data found.")
            return

        data = preprocess_data(users, courses)
        user_vectors, course_vectors = data['userVectors'], data['courseVectors']

        mock_data = generate_sync_data(user_vectors, course_vectors)
        xs_users, xs_courses, ys = mock_data['xsUsers'], mock_data['xsCourses'], mock_data['ys']

        # # Train the model
        # model = build_foundation_model(xs_users, xs_courses, 1e-4)
        # results = await train_model(model, xs_users, xs_courses, ys)

        # # Show training chart result
        # plot_training_history(results)

        # Retrain the model after saving
        resultReloaded = await retrain_model(xs_users, xs_courses, ys, 1e-3)

        # Show re-training chart result
        plot_training_history(resultReloaded)

        # Example recommendation for first user
        recommendations = recommend_courses(model, user_vectors[1]['vector'], course_vectors)
        print("Top 5 recommendations for user 1:", recommendations)

    except Exception as e:
        print(f"Error in main: {e}")

# Run the script
import asyncio
asyncio.run(main())
