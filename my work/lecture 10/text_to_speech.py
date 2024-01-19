# import jiwer  # you may need to install this library

# def mean_wer(solution, submission):
#     joined = solution.merge(submission.rename(columns={'sentence': 'predicted'}))
#     domain_scores = joined.groupby('domain').apply(
#         # note that jiwer.wer computes a weighted average wer by default when given lists of strings
#         lambda df: jiwer.wer(df['sentence'].to_list(), df['predicted'].to_list()),
#     )
#     return domain_scores.mean()

# assert (solution.columns == ['id', 'domain', 'sentence']).all()
# assert (submission.columns == ['id',' sentence']).all()


# from transformers import DetrImageProcessor, DetrForObjectDetection
# import torch
# from PIL import Image
# import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)

# # convert outputs (bounding boxes and class logits) to COCO API
# # let's only keep detections with score > 0.9
# target_sizes = torch.tensor([image.size[::-1]])
# results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#             f"Detected {model.config.id2label[label.item()]} with confidence "
#             f"{round(score.item(), 3)} at location {box}"
#     )

import transformers

pipeline = transformers.pipeline("text-classification")

text = "This is a news article about the latest COVID-19 outbreak."

results = pipeline(text)

print(results)


