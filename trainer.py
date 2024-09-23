import os
import random
import matplotlib.pyplot as plt
from datetime import datetime

# Full integer list with categories
exercise_categories = {
    "Select": [1757, 584, 595, 1148, 1683],
    "Basic Joins": [197, 1661, 577, 1280, 570, 1934],
    "Basic Aggregation Functions": [620, 1251, 1075, 1633, 1211, 1193],
    "Sorting and Grouping": [2356, 1141, 1070, 596, 1729, 619, 1045],
    "Advanced Select and Joins": [1731, 1789, 610, 180, 1164, 1204, 1907],
    "Subqueries": [1978, 626, 1341, 1321, 602, 585, 185],
    "Advanced String Functions": [1667, 1527, 196, 176, 1484, 1327, 1517]
}

# Flattening the list of integers and storing category info
integer_list = []
integer_category_map = {}

for category, integers in exercise_categories.items():
    integer_list.extend(integers)
    for num in integers:
        integer_category_map[num] = category

# Sample flashcard questions and answers with categories
flashcards = [
    {"question": r"What is the Law of Large Numbers?", "answer": r"\bar{X}_n \rightarrow \mu$ as $n \rightarrow \infty$.", "subject": "Statistics"},
    {"question": r"What is the Central Limit Theorem?", "answer": r"Let $x_1,...,x_n$ be a random sample of $n$ observations of a random variable with expected value $\mu$ and finite variance $\sigma^2$. Then as $n \rightarrow \infty$, ($\bar{X}_n - \mu) / \sigma_{\bar{X}_n} \sim $ N(0,1) where $\sigma_{\bar{X}_n} = \frac{\sigma}{\sqrt{n}}$", "subject": "Statistics"},
    {"question": r"What is a confidence interval?", "answer": r"A confidence interval is a range of values which under repeated observation would contain a true parameter value (confidence level)% of the time.", "subject": "Statistics"},
    {"question" : r"Pitfalls of A/B testing", "answer": r"Imbalanced data/allocation to treatment groups, low sample size, temporal effects, hidden treatments and confounders", "subject": r"Statistics"},
    {"question" : r"What is covariance", "answer": r"Cov$(X,Y) = E(XY) - E(X)E(Y)$, measures the direction and strength of the linear relationship between X and Y.", "subject": r"Statistics"},
    {"question" : r"Correlation", "answer": r"Corr$(X,Y) = \frac{Cov(X,Y)}{\sigma_x \sigma_y}$ ; Correlation is normalized covariance and measures the direction and strength of the linear relationship between X and Y", "subject": r"Statistics"},
    {"question" : r"Type I Error", "answer": r"False positive. Rejecting the null hypothesis when it is true. $P($false positive$) = \alpha$", "subject": r"Statistics"},
    {"question" : r"Type II Error", "answer": r"False negative. Failing to reject a null hypothesis when it is false. P(false negative) = $\beta$", "subject": r"Statistics"},
    {"question" : r"What is Power?", "answer": r"The power of a test is the probability of rejecting the null hypothesis, when it is false. Power = $1 - \beta$", "subject": r"Statistics"},
    {"question" : r"What are the assumptions of a t-test?", "answer": r"Independence of observations, normality of data", "subject": r"Statistics"},
    {"question" : r"Wilcoxon Signed-rank Test", "answer": r"Nonparametric test to see whether their population mean ranks differ. Also tests for nonzero median.", "subject": r"Statistics"},
    {"question" : r"Kolmogorov-Smirnov Test", "answer": r"Tests whether a sample came from a given reference distribution, or if two samples came from the same distribution.", "subject": r"Statistics"},
    {"question" : r"Durbin-Watson Test", "answer": r"Tests for the presence of autocorrelation in the residuals of a regression analysis.", "subject": r"Statistics"},
    {"question" : r"Why do we care about autocorrelation?", "answer": r"Autocorrelation violates the assumption of independence in linear regression, leading to biased estimates.", "subject": r"Statistics"},
    {"question" : r"What are the assumptions of linear regression?", "answer": r"Linear relationship between predictors and response, independence of observations, constant variance (residuals vs. fitted values), normality of residuals, no multicollinearity", "subject": r"Statistics"},
    {"question" : r"Linear regression assumptions by importance", "answer": r"1. Linearity: super wrong means super biased estimates. 2. Dependence of errors: strong dependence means less info in data than sample size suggests. 3. Nonconstant variance: failures to identify could make inaccurate inferences, in particular the quantification of uncertainty. 4. Normality- large datasets will be robust to lack of normality by CLT", "subject": r"Statistics"},
    {"question" : r"Unbiased Estimator", "answer": r"$E(\hat{\theta}) = \theta", "subject": r"Statistics"},
    {"question" : r"Consistent Estimator", "answer": r"$\hat{\theta}_n$ converges in probability to $\theta$ as $n \rightarrow \infty$", "subject": r"Statistics"},
    {"question" : r"Independence vs Uncorrelatedness", "answer": r"Independence implies uncorrelatedness, uncorrelatedness does not imply independence", "subject": r"Statistics"},
    
    {"question" : r"How do you approach imbalanced classes in binary classification?", "answer": r"Get more data, oversample the rare class while undersampling the common class, generate synthetic data with a resampling technique like SMOTE, harshly penalize incorrect classification of rare class", "subject": r"Machine Learning"},
    {"question" : r"Mean Squared Error vs. Mean Absolute Error", "answer": r"MSE gives more weight to outliers and large errors. Use when outliers have meaning or with lienar models. MAE is less sensitive to outliers, more robust to noisy error. Use if outliers are common and not informative, robustness to noise is critical, or with non-normal error.", "subject": r"Machine Learning"},

    {"question" : r"What is precision and recall?", "answer": r"Precision = $\frac{TP}{TP + FP}$, Recall = $\frac{TP}{TP + FN}$", "subject": r"Machine Learning"},
    {"question" : r"What is precision?", "answer": r"Accuracy of positive predictions, the ratio of true positives over all positive predictions", "subject": r"Machine Learning"},
    {"question" : r"What is recall?", "answer": r"Ratio of correctly predicted positives to all actual positives", "subject": r"Machine Learning"},

    {"question" : r"What is the ROC curve?", "answer": r"", "subject": r""},
    {"question" : r"How do you interpret an AUC score?", "answer": r"", "subject": r""},

    {"question" : r"How do you choose $k$ in $k$-means clustering?", "answer": r"Elbow method: run $k$-means for a range of $k$ values. For each, calculate the sum of squared error and plot as a function of $k$. Locate the 'elbow' where the rate of decline in SSE shifts.", "subject": r"Machine Learning"},

    {"question" : r"How do you make models more robust to outliers?", "answer": r"First, investigate for meaning. Then detect and remove (high z-score). Transform data to reduce the influence of extreme values. Apply regularization techniques. Use a robust loss function like Huber or Log-Cosh loss, behaves like MSE for small errors and MAE for large errors. Use a model robust to outliers, like an ensemble method/ random forest.", "subject": r"Machine Learning"},
    {"question" : r"How do you detect outliers?", "answer": r"Easy way: high Z-score", "subject": r"Machine Learning"},
    {"question" : r"How do you deal with multicollinearity?", "answer": r"Remove or combine predictors, or use domain knowledge.", "subject": r"Statistics"},

    {"question" : r"Why does multicollinearity matter?", "answer": r"Multicollinearity can produce inaccurate p-values and confidence intervals, significant predictors may appear insignificant. ", "subject": r"Statistics"},

    {"question" : r"How do random forests improve upon decision trees?", "answer": r"Decision trees are prone to overfitting. RF are less susceptible since they average multiple trees trained on different parts of a dataset. They improve accuracy by averaging errors, and reduce variance caused by feature selection. They decrease the effect of noise and outliers, reducing variance while not increasing bias.", "subject": r"Machine Learning"},
    {"question" : r"How do you deal with missing data?", "answer": r"First ask what is missing- is there a pattern? Is missing data a problem? Then consider imputing missing data if doing so increases model performance. Consider a model that can deal with missing data, like a random forest. Can you source data from a third party?", "subject": r"Machine Learning"},
    {"question" : r"What is Gradient Boosting?", "answer": r"Builds trees sequentially, each tree trained to correct the errors of previous trees. Prone to overfitting. Use when optimal performance is necessary.", "subject": r"Machine Learning"},
    {"question" : r"How do you know if you have enough data?", "answer": r"Fit models using a subset of the data. Are the results significantly worse, or does the improvement caused by having more data begin to offer diminishing returns? (Use $R^2$, SSE, etc)", "subject": r"Machine Learning"},
    {"question" : r"What is the Bias-Variance Tradeoff?", "answer": r"MSE = E$(Y-\hat{f}(x))^2$ = Bias$^2$ + Var + $\sigma^2$.", "subject": r"Machine Learning"},
    {"question" : r"How does high bias and high variance affect models?", "answer": r"High bias ignores training data, underfitting. High variance pays too much attention to the data, overfitting. ", "subject": r"Machine Learning"},
    {"question" : r"What is $k$-fold cross validation? Why do you do it?", "answer": r"Evaluate the model, avoid overfitting, assess robustness (how well does the  model perform on different variations of the data?)", "subject": r"Machine Learning"},
    {"question" : r"How do you perform $k$-fold cross validation?", "answer": r"Divide the data into $k$ nonoverlapping, equally sized subsets. For each subset, train them on the remaining $k-1$ subsets and evaluate performance against the chosen subset. Evaluate metrics by combining them.", "subject": r"Machine Learning"},
    {"question" : r"What does Entropy measure?", "answer": r"Entropy measures the impurity or randomness in data. In a decision tree, it measures heterogeneity at each node.", "subject": r"Machine Learning"},
    {"question" : r"What is Stochastic Gradient Descent?", "answer": r"Updates parameters using a random subset of the data, reducing compute costs and preventing entrapment into local minima.", "subject": r"Machine Learning"},


    {"question" : r"What are the assumptions of an ARIMA model?", "answer": r"", "subject": r"Statistics"},

    {"question" : r"How do you deploy an AI model?", "answer": r"", "subject": r"Machine Learning"},
    {"question" : r"What are some model validation strategies?", "answer": r"", "subject": r"Machine Learning"},
    {"question" : r"What is L1 regularization?", "answer": r"Lasso. Adds a penalty equal to the absolute value of the magnitude of the coefficient. Leads to sparse solutions, so more coefficients are exactly zero. Good for feature selection. Leads to non-differentiable points, more difficult optimization. Simpler models, better for high dimensional data.", "subject": r"Machine Learning"},
    {"question" : r"What is L2 regularization?", "answer": r"Ridge. Adds a penalty equal to the square of the magnitude of the coefficients. Promotes coefficients being very small, distributes error but does not promote sparse solutions. Easier to optimize than L1 regularization. ", "subject": r"Machine Learning"},
    
    {"question" : r"How do you test for Normality in data?", "answer": r"", "subject": r"Statistics"},

    {"question" : r"When would you use an SVM?", "answer": r"", "subject": r"Machine Learning"},
    {"question" : r"Describe the Kernel Trick for SVMs", "answer": r"", "subject": r"Machine Learning"},
    {"question" : r"When should you NOT use a random forest?", "answer": r"", "subject": r"Machine Learning"},
    {"question" : r"How do you perform Principal Components Analysis?", "answer": r"", "subject": r"Machine Learning"},
    {"question" : r"What model evaluation strategies should you use for binary classification?", "answer": r"", "subject": r"Machine Learning"},
    {"question" : r"What is Specificity and Sensitivity?", "answer": r"", "subject": r""},


    {"question" : r"What is a LEFT JOIN?", "answer": r"", "subject": r"SQL"},
    {"question" : r"What is an INNER JOIN?", "answer": r"", "subject": r"SQL"},

    {"question" : r"How does Survival Analysis arise? What is a business application of survival analysis?", "answer": r"The analysis of the time until an event occurs. A business application of survival analysis is modeling user churn, the process by which customers cancel subscription to a service.", "subject": r"Statistics"}, 

    #TODO: ADD TIME SERIES QUESTIONS

    #{"question" : r"", "answer": r"", "subject": r""},
    #{"question" : r"", "answer": r"", "subject": r""},
    # Add more flashcards with their subjects
]

# Function to create a new directory based on the current timestamp
def create_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"flashcards_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

# Function to generate flashcard images using matplotlib
def generate_flashcards(folder_name, selected_flashcards):
    for i, flashcard in enumerate(selected_flashcards, start=1):
        # Create the question image
        plt.figure(figsize=(8, 6))
        question_text = f"Subject: {flashcard['subject']}\nQ: {flashcard['question']}"
        plt.text(0.5, 0.5, question_text, fontsize=20, ha='center', va='center', wrap=True)
        plt.axis('off')  # Hide axes
        question_filename = os.path.join(folder_name, f"flashcard_{i}_question.png")
        plt.savefig(question_filename, bbox_inches='tight', dpi=300)
        plt.close()

        # Create the answer image
        plt.figure(figsize=(8, 6))
        answer_text = f"Subject: {flashcard['subject']}\nA: {flashcard['answer']}"
        plt.text(0.5, 0.5, answer_text, fontsize=20, ha='center', va='center', wrap=True)
        plt.axis('off')  # Hide axes
        answer_filename = os.path.join(folder_name, f"flashcard_{i}_solution.png")
        plt.savefig(answer_filename, bbox_inches='tight', dpi=300)
        plt.close()

    print(f"Flashcards saved in folder: {folder_name}")

# Function to draw N values from the list of integers
def draw_n_integers(N):
    return random.sample(integer_list, N)

# Function to draw M flashcards from the list
def draw_m_flashcards(M):
    return random.sample(flashcards, M)

# Function to create the "DailySQL.png" image with categories
def create_daily_sql_image(folder_name, drawn_integers):
    # Create the message showing the categories and corresponding integers
    category_message = ""
    categorized_exercises = {}

    for num in drawn_integers:
        category = integer_category_map[num]
        if category not in categorized_exercises:
            categorized_exercises[category] = []
        categorized_exercises[category].append(num)

    for category, numbers in categorized_exercises.items():
        category_message += f"{category}: {', '.join(map(str, numbers))}\n"

    # Create the image using matplotlib
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, f"Complete the following exercises:\n{category_message}", fontsize=20, ha='center', va='center', wrap=True)
    plt.axis('off')  # Hide axes
    daily_sql_filename = os.path.join(folder_name, "DailySQL.png")
    plt.savefig(daily_sql_filename, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"DailySQL image saved as: {daily_sql_filename}")

# Prompt the user for N and M
def prompt_for_values():
    # Ask the user how many integers (N) they want to draw
    N = int(input(f"How many integers would you like to draw (1-{len(integer_list)})? "))
    while N < 1 or N > len(integer_list):
        N = int(input(f"Please enter a valid number of integers (1-{len(integer_list)}): "))

    # Ask the user how many flashcards (M) they want to generate
    M = int(input(f"How many flashcards would you like to generate (1-{len(flashcards)})? "))
    while M < 1 or M > len(flashcards):
        M = int(input(f"Please enter a valid number of flashcards (1-{len(flashcards)}): "))

    return N, M

# Main logic
def main():
    # Prompt for user input (N and M)
    N, M = prompt_for_values()

    # Draw N integers and print them
    drawn_integers = draw_n_integers(N)
    print(f"Drawn Integers: {drawn_integers}")

    # Draw M flashcards
    selected_flashcards = draw_m_flashcards(M)

    # Create a folder to save flashcards
    folder_name = create_directory()

    # Generate flashcards and save them as images
    generate_flashcards(folder_name, selected_flashcards)

    # Create the "DailySQL.png" image
    create_daily_sql_image(folder_name, drawn_integers)

# Run the main function
if __name__ == "__main__":
    main()
