import base64
import requests
import os
import json
import argparse


def parse_score(review):
    try:
        score = review.split('\n')[0]
        score = score.replace(' ', '')
        return float(score)
    except Exception as e:
        print(e)
        print('error', review)
        return -1


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_eval(args, image_content, text_content):

    payload = {
        "model": "gpt-4-turbo-2024-04-09",  # "gpt-4-turbo-2024-04-09", "gpt-4.1-2025-04-14"
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful and precise assistant for checking the quality of the AI assistant report based on the reference report."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_content}"
                        }
                    },
                    {
                        "type": "text",
                        "text": text_content,
                    }
                ]
            }
        ],
        "max_tokens": args.max_tokens
    }

    response = requests.post(args.BASE_URL, headers=headers, json=payload)
    res = response.json()
    # conversations = res['choices'][0]['message']["content"]

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-based report evaluation.')
    parser.add_argument("--API_SECRET_KEY", type=str,
                        default="",
                        help="api secret key")
    parser.add_argument("--BASE_URL", type=str,
                        default="",
                        help="base url")
    parser.add_argument("--image_path", type=str,
                        default="./test_set/UW-Report(partial)/JPEGImages",
                        help="path to test images")
    parser.add_argument("--json_ReferenceReport_path", type=str,
                        default="./test_set/UW-Report(partial)/ReferenceReport",
                        help="path to GT reference report json files")
    parser.add_argument("--json_AssistantReport_path", type=str,
                        default="./results/textual_report",
                        help="path to assistant report json files")   # AI报告位置
    parser.add_argument("--json_scoring_path", type=str,
                        default="./results/scoring_report",
                        help="path to save scoring json files")   # 评分报告存储位置
    parser.add_argument('--max_tokens', type=int, default=512, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.API_SECRET_KEY}"
    }

    prompt = "We would like to request your feedback on the performance of an AI assistant in generating a waterlogging assessment report for the provided image. \nPlease rate the accuracy, comprehensiveness, and level of detail of the report from the AI assistant based on the reference report. The AI assistant receives an overall score on a scale of 1 to 10. If it makes a wrong judgment on whether there is waterlogging in the image, give it 1 point. If the judgment is correct, give it 2 to 10 points, with higher scores indicating better overall performance. It should be emphasized that the reference report is reliable and can accurately reflect the actual waterlogging situation shown in the image. \nPlease first output a single line containing only one value indicating the score for the AI assistant. \nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the reports were presented does not affect your judgment."

    score_record = []
    i = 0
    for image_name in os.listdir(args.image_path):

        json_name = image_name.replace('.jpg', '.json')

        if os.path.exists(os.path.join(args.json_scoring_path, json_name)):
            continue

        base64_image = encode_image(os.path.join(args.image_path, image_name))  # Getting the base64 string

        with open(os.path.join(args.json_ReferenceReport_path, json_name), 'r') as file:
            reference_report = json.load(file)["report"]

        with open(os.path.join(args.json_AssistantReport_path, json_name), 'r') as file:
            assistant_report = json.load(file)["report"][0]["value"]

        text_content = f"[Reference Report]\n{reference_report}\n\n[End of Reference Report]\n\n[AI Assistant Report]\n{assistant_report}\n\n[End of AI Assistant Report]\n\n[System]\n{prompt}\n\n"

        eval_response = get_eval(args, base64_image, text_content)
        with open(os.path.join(args.json_scoring_path, json_name), 'w') as file:
            json.dump(eval_response, file, indent=2)

        score_record.append(parse_score(eval_response['choices'][0]['message']["content"]))
        print(i)
        i += 1

    print(f"Number: {len(score_record)}\nAverage Score: {sum(score_record)/len(score_record)}\n{score_record}")
