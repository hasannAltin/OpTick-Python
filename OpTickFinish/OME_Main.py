from flask import Flask, request, jsonify
from PIL import Image, ExifTags
import cv2
import numpy as np
import utlis
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

widthImg, heightImg = 900, 900
choices = 5
questions = None
answer_key = None
marked_optic = None

def process_answer_key(image_stream):
    img = Image.open(image_stream)
    try:
        exif = img._getexif()
        if exif is not None:

            for tag, value in exif.items():
                if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                    orientation = value
                    break
            else:
                orientation = 1
    except (AttributeError, KeyError, IndexError):

        orientation = 1


    if orientation == 3:
        img = img.rotate(180, expand=True)
    elif orientation == 6:
        img = img.rotate(270, expand=True)
    elif orientation == 8:
        img = img.rotate(90, expand=True)

    img = np.array(img)
    img = cv2.resize(img, (widthImg, heightImg))

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectCon = utlis.rectCountour(contours)
    if len(rectCon) >= 3:
        imgWarpColoredList = []

        for i in range(2):
            biggestContour = utlis.getCornerPoints(rectCon[i])
            if biggestContour.size != 0:
                biggestContour = utlis.reorder(biggestContour)
                pt1 = np.float32(biggestContour)
                pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                matrix = cv2.getPerspectiveTransform(pt1, pt2)
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
                imgWarpColoredList.append(imgWarpColored)

        if len(imgWarpColoredList) == 2:
            final_image = np.hstack((imgWarpColoredList[0], imgWarpColoredList[1]))

            imgWarpGray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]

            boxes = utlis.splitBoxes(imgThresh,questions)
            myPixelVal = np.zeros((questions, choices))
            countC = 0
            countR = 0

            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countR += 1
                    countC = 0

            myIndex = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                max_val = np.amax(arr)
                min_val = np.amin(arr)
                pixel_difference = max_val - min_val

                treshold = 0.37
                if pixel_difference < (max_val * treshold):
                    myIndex.append(-1)
                else:
                    myIndexVal = np.where(arr == max_val)
                    myIndex.append(int(myIndexVal[0][0]))

            return myIndex

        return None


##############################################################
def process_marked_optic(image_stream):
    return process_answer_key(image_stream)

def clear_answer_key():
    global answer_key
    answer_key = None


def clear_marked_optic():
    global marked_optic
    marked_optic = None

@app.route('/number_of_questions', methods=['POST'])
def number_of_questions():
    global questions
    data = request.get_json()
    if 'questions' not in data:
        return jsonify({"error": "Questions missing"}), 400
    questions = int(data['questions'])

    return jsonify({"message": "Question count received successfully", "questions": questions})


@app.route('/upload-answer-key', methods=['POST'])
def upload_answer_key():
    global answer_key
    clear_answer_key()

    if 'answer_key' not in request.files:
        return jsonify({"error": "Answer key photo is missing"}), 400

    photo = request.files['answer_key']
    if photo.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        answer_key = process_answer_key(photo.stream)
        if answer_key:
            return jsonify({"message": "Answer key processed successfully", "answer_key": answer_key})
        return jsonify({"error": "Failed to process answer key"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/grade-marked-optic', methods=['POST'])
def grade_marked_optic():
    if answer_key is None:
        return jsonify({"error": "Answer key not uploaded yet"}), 400

    if 'marked_optic' not in request.files:
        return jsonify({"error": "Marked optic photo is missing"}), 400

    clear_marked_optic()

    image_file = request.files['marked_optic']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    marked_optic = process_marked_optic(image_file.stream)
    if marked_optic:
        grading = []
        for i in range(len(answer_key)):
            if marked_optic[i] == -1:
                grading.append(0)
            else:
                grading.append(1 if answer_key[i] == marked_optic[i] else 0)

        score = (sum(grading) / len(answer_key)) * 100
        return jsonify({"score": score})

    return jsonify({"error": "Failed to process marked optic"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
