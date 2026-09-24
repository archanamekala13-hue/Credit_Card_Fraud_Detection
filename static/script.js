const thresholdSlider =
    document.getElementById("threshold");


const thresholdValue =
    document.getElementById("thresholdValue");


const accuracy =
    document.getElementById("accuracy");


const precision =
    document.getElementById("precision");


const recall =
    document.getElementById("recall");


const f1 =
    document.getElementById("f1");


const tn =
    document.getElementById("tn");


const fp =
    document.getElementById("fp");


const fn =
    document.getElementById("fn");


const tp =
    document.getElementById("tp");



async function evaluateThreshold() {


    const threshold =
        Number(
            thresholdSlider.value
        );


    thresholdValue.textContent =
        threshold.toFixed(2);


    try {


        const response =
            await fetch(
                `/api/evaluate?threshold=${threshold}`
            );


        if (!response.ok) {

            throw new Error(
                "Unable to get prediction results."
            );

        }


        const data =
            await response.json();


        accuracy.textContent =
            data.accuracy + "%";


        precision.textContent =
            data.precision + "%";


        recall.textContent =
            data.recall + "%";


        f1.textContent =
            data.f1_score + "%";


        const matrix =
            data.confusion_matrix;


        /*
            Confusion matrix:

            [[TN, FP],
             [FN, TP]]
        */


        tn.textContent =
            matrix[0][0];


        fp.textContent =
            matrix[0][1];


        fn.textContent =
            matrix[1][0];


        tp.textContent =
            matrix[1][1];


    }

    catch (error) {

        console.error(error);

        accuracy.textContent = "--";

        precision.textContent = "--";

        recall.textContent = "--";

        f1.textContent = "--";

    }

}



thresholdSlider.addEventListener(
    "input",
    evaluateThreshold
);


// Load initial results
evaluateThreshold();