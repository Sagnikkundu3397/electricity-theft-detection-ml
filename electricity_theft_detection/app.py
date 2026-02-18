import os
import sys

from flask import Flask, request, render_template

# Add the project root to sys.path to ensure src is discoverable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# noqa: E402
from src.pipeline.predict_pipeline import PredictPipeline, CustomData  # noqa: E402, E501
from src.pipeline.train_pipeline import TrainPipeline  # noqa: E402

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["GET"])
def train_route():
    try:
        pipeline = TrainPipeline()
        best_name, best_metrics, full_report = pipeline.run()
        return render_template(
            "train_result.html",
            best_model=best_name,
            metrics=best_metrics,
            report=full_report,
        )
    except Exception as e:
        return f"<h3>Training Failed</h3><p>{str(e)}</p>", 500


@app.route("/predict", methods=["GET", "POST"])
def predict_route():
    if request.method == "GET":
        return render_template("predict.html")

    try:
        # Extra check for form values
        mean = request.form.get("mean")
        std = request.form.get("std")
        min_val = request.form.get("min")
        max_val = request.form.get("max")
        zeros = request.form.get("zeros")

        if not all([mean, std, min_val, max_val, zeros]):
            return render_template("predict.html",
                                   error="Error: All fields are required")

        data = CustomData(
            mean=float(mean),
            std=float(std),
            min=float(min_val),
            max=float(max_val),
            zeros=int(zeros)
        )

        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        prediction, probability = pipeline.predict(df)

        result_text = ("⚠️ ELECTRICITY THEFT DETECTED" if prediction[0] == 1
                       else "✅ NORMAL CONSUMER")
        res_class = "theft" if prediction[0] == 1 else "normal"
        confidence = round(probability[0] * 100, 2)

        return render_template(
            "predict.html",
            result=result_text,
            res_class=res_class,
            confidence=confidence,
            prediction=prediction[0]
        )

    except Exception as e:
        return render_template("predict.html", error=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
