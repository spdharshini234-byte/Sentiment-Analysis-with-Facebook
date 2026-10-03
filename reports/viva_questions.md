# Viva Questions and Answers - Sentiment Analysis with Facebook

**Q1. Why Facebook sentiment analysis?**  
Facebook is where people openly share opinions. Knowing whether they are positive or negative helps businesses react faster.

**Q2. Why this dataset?**  
It was provided by our course for this project. It has text, a sentiment label, a category and a date, which lets us build a classifier and a trend dashboard.

**Q3. What is NLP?**  
Natural Language Processing is making computers work with human language, for example cleaning text and turning it into numbers.

**Q4. What is sentiment analysis?**  
Deciding whether a piece of text is positive, negative or neutral.

**Q5. Why TF-IDF?**  
It turns text into numbers and gives more weight to words that are important in a post but not common in every post.

**Q6. Why Naive Bayes?**  
It is fast, simple and works well on small text datasets. It was our best model on the validation set.

**Q7. Why Logistic Regression?**  
It is a strong, simple linear model that gives probabilities and is easy to explain.

**Q8. Why SVM?**  
A Linear SVM finds the best separating line between classes and often works very well on TF-IDF features.

**Q9. Why compare models?**  
No model is best on every dataset. Comparing them on the same split lets us choose fairly.

**Q10. Why macro-F1?**  
Our classes are imbalanced. Macro-F1 averages the F1 of each class equally, so a poor result on the small negative class is not hidden.

**Q11. What is overfitting?**  
When a model memorises training data and does badly on new data. We check by comparing train, validation and test scores.

**Q12. What is data leakage?**  
When information from the test set leaks into training so scores look better than they really are. Repeated sentences cause this, so we split by sentence group.

**Q13. Why train, validation and test?**  
Train to learn, validation to choose the model and settings, test to give one honest final score.

**Q14. What is FastAPI?**  
A Python framework for building web APIs. It validates input automatically and creates documentation at /docs.

**Q15. Why React?**  
It builds interactive dashboards from reusable components, and Recharts gives ready-made charts.

**Q16. How does the frontend talk to the backend?**  
The browser sends HTTP requests (fetch) to the /api endpoints and receives JSON, which React shows as cards and charts.

**Q17. What happens when a comment is entered?**  
The text goes to POST /api/analyze, the saved pipeline cleans it, TF-IDF converts it, the classifier returns probabilities, and the UI shows the label, confidence and important words.

**Q18. How is confidence calculated?**  
It is the model's probability for the predicted class. We call it high at 0.70 or more, medium from 0.50, and low below 0.50.

**Q19. How are dashboard values calculated?**  
On start-up the saved model scores every prepared post into SQLite. The API counts and aggregates those rows. Nothing is typed in by hand.

**Q20. How is deployment done?**  
Backend on Render with the start command uvicorn backend.app.main:app; frontend on Vercel with VITE_API_URL pointing at the Render URL. Details are in the README.

**Q21. How did you handle duplicates?**  
Exact duplicate records: 0. But 3,361 texts repeat with different ids and dates, so we kept them and grouped identical texts to prevent leakage.

**Q22. Why is the accuracy not higher?**  
The same sentence often has different labels in the dataset (659 of 1,639 unique texts carry more than one label, so text-only accuracy cannot exceed about 70.08%). That sets a ceiling for any text-based model.

**Q23. Did you change the original dataset?**  
No. The raw CSV stays in data/raw. Cleaned data is a separate file in data/processed.

**Q24. What is the baseline and why use it?**  
A model that always predicts the most common label (macro-F1 0.207). Our model must beat it to be useful.

**Q25. Why keep words like 'not' and emojis?**  
'Not good' and 'good' have opposite meanings, and emojis show feeling. Removing them would hurt sentiment.

**Q26. What is cross-validation?**  
Splitting training data into folds, training on some and checking on the rest, to get a more stable score when tuning settings.

**Q27. What is GridSearchCV?**  
It tries every combination of chosen settings with cross-validation and keeps the best one.

**Q28. What is a confusion matrix?**  
A table of actual versus predicted labels. The diagonal shows correct predictions.

**Q29. How is Sentiment Health calculated?**  
100 x (positive + 0.5 x neutral) / total posts. Neutral counts as half.

**Q30. How is the Customer Voice Index calculated?**  
100 x (positive - negative) / (positive + negative). It ranges from -100 to +100.

**Q31. Why is the date trend available?**  
The dataset has a valid timestamp column. If a dataset had none, the app would show a message and never invent dates.

**Q32. Does the project use live Facebook data?**  
No. It analyses the supplied dataset. Live collection is listed as future scope.

**Q33. What are the limitations?**  
Inconsistent labels, bag-of-words cannot catch sarcasm, fixed dataset, unreliable language column.

**Q34. How can the project be improved?**  
Better labels, a transformer model, real language detection, emotion labels, and scheduled retraining.

**Q35. How do you make sure the model file and the API match?**  
The pipeline includes the cleaning step, and tests check that the saved labels match the model's classes.

**Q36. Which nine phases did you follow?**  
Problem Definition, Requirement Analysis, Data Acquisition, Data Preparation, Exploratory Data Analysis (EDA), Model Selection, Model Development, Model Evaluation, Solution Integration & Testing.
