# aws-certified-ai-practitioner
* Artificial Intelligence is a field of computer science dedicated to solving problems that we commonly associate with human intelligence.
* Artificial Intelligence >  Machine Learning > Deep Learning > Generative AI.
* Generative AI is used to generate new data that is similar to the data it was trained on. To generate data, we must rely on a Foundation Model. Foundation Models are trained on a wide variety of input data. The models may cost tens of millions of dollars to train.
* Large Language Models (LLM) is the type of AI designed to generate coherent human-like text (Chat GPT).
* Non-deterministic: the generated text may be different for every user that uses the same prompt.
* Amazon Bedrock
  * Build Generative AI (Gen-AI) applications on AWS.
  * Fully-managed service, no servers for you to manage.
  * Keep control of your data used to train the model.
  * Access to a wide range of Foundation Models (FM)
*  Amazon Titan
  * High-performing Foundation Models from AWS.
  * Can be customized with your own data.
* Automatic Evaluation vs Human Evaluation.
*  Business Metrics to Evaluate a Model On: User Satisfaction, Average Revenue Per User (ARPU), Cross-Domain Performance, Conversion Rate and Efficiency.
*  RAG = Retrieval-Augmented Generation
  * Allows a Foundation Model to reference a data source outside of its training data.
* Amazon Bedrock – Guardrails
  * Control the interaction between users and Foundation Models (FMs).
  * Filter undesirable and harmful content.
  * Remove Personally Identifiable Information (PII).
  * Reduce hallucinations
* Prompt Engineering = developing, designing, and optimizing prompts to enhance the output of FMs for your needs.
* Negative Prompting is a technique where you explicitly instruct the model on what not to include or do in its response.
* Zero-Shot Prompting - Present a task to the model without providing examples or explicit training for that specific task.
* Few-Shots Prompting - Provide examples of a task to the model to guide its output.
* Chain of Thought Prompting - Divide the task into a sequence of reasoning steps, leading to more structure and coherence.
* Retrieval-Augmented Generation (RAG) - Combine the model’s capability with external data sources to generate a more informed and contextually rich response.
* Amazon Q Business- Fully managed Gen-AI assistant for your employees. Based on your company’s knowledge and data.
* Amazon Q Apps - Create Gen AI-powered apps without coding by using natural language.
* Amazon Q Developer - Answer questions about the AWS documentation and AWS service selection. Answer questions about resources in your AWS account.
* Deep Learning - Uses neurons and synapses (like our brain) to train a model.
* Supervised Learning
  * Learn a mapping function that can predict the output for new unseen input data.
  * Needs labeled data: very powerful, but difficult to perform on millions of datapoints.
  * Regression - Used to predict a numeric value based on input data.
  * Classification - Used to predict the categorical label of input data.
* Feature Engineering - The process of using domain knowledge to select and transform raw data into meaningful features.
* Unsupervised Learning - The goal is to discover inherent patterns, structures, or relationships within the input data.
* Reinforcement Learning - A type of Machine Learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative rewards.
* • Inferencing is when a model is making prediction on new data.
* AWS AI Services are pre-trained ML services for your use case.
* Amazon Comprehend - Uses machine learning to find insights and relationships in text.
* Amazon Translate.
* Amazon Transcribe.
* Amazon Polly - Turn text into lifelike speech using deep learning.
* Amazon Rekognition - Find objects, people, text, scenes in images and videos using M.
* Amazon Forecast - Fully managed service that uses ML to deliver highly accurate forecasts.
* Amazon Lex & Connect - same technology that powers Alexa. Receive calls, create contact flows, cloud-based virtual contact center.
* Amazon Personalize - Fully managed ML-service to build apps with real-time personalized recommendations.
* Amazon Textract - Automatically extracts text, handwriting, and data from any scanned documents using AI and ML.
* Amazon Kendra - Fully managed document search service powered by Machine Learning.
* Amazon Mechanical Turk - Crowdsourcing marketplace to perform simple human tasks.
* Amazon Augmented AI (A2I) - Human oversight of Machine Learning predictions in production.
* AWS DeepRacer.
* Amazon Transcribe Medical - Automatically convert medical-related speech to text.
* Amazon Comprehend Medical - Amazon Comprehend Medical detects and returns useful information in unstructured clinical text.
* Amazon SageMaker - Fully managed service for developers / data scientists to build ML models.
* SageMaker Clarify - Evaluate Foundation Models. Ability to detect and explain biases in your datasets and models
* SageMaker Canvas - Build ML models using a visual interface (no coding required).
* SageMaker Automatic Model Tuning: tune hyperparameters
* SageMaker Deployment & Inference: real-time, serverless, batch, async
* SageMaker Studio: unified interface for SageMaker
* SageMaker Data Wrangler: explore and prepare datasets, create features
* SageMaker Feature Store: store features metadata in a central place
* SageMaker Clarify: compare models, explain model outputs, detect bias
* SageMaker Ground Truth: RLHF, humans for model grading and data labeling
* SageMaker Model Cards: ML model documentation 
* SageMaker Model Dashboard: view all your models in one place 
* SageMaker Model Monitor: monitoring and alerts for your model
* SageMaker Role Manager: access control
* SageMaker JumpStart: ML model hub & pre-built ML solutions
* SageMaker Canvas: no-code interface for SageMaker
* MLFlow on SageMaker: use MLFlow tracking servers on AWS
