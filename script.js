import { CLASSES } from './labels.js'

const stopBtn = document.getElementById("stopBtn");
const flipBtn = document.getElementById("flipBtn");


/* Activating a Webcam */
let constraints = {
    audio: false,
    video: {
        facingMode: "environment"
    }
}
async function setupWebcam(videoRef) {
    // let allMediaDevices = navigator.mediaDevices
    // flip button element

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    //     let front = false;
    //         document.getElementById("flip-button").onclick = () => {
    //         front = !front;
    //     };
        
        const webcamStream = await navigator.mediaDevices.getUserMedia(constraints)
        
        // This conditional check is to support older browsers that do not 
        // support the new srcObject configuration. This can likely be        
        // deprecated depending on your support needs.
        if ('srcObject' in videoRef) {
            videoRef.srcObject = webcamStream
        } else {
            videoRef.src = window.URL.createObjectURL(webcamStream)
        }

        // You can’t access the video until it is loaded, so the event is 
        // wrapped in a promise so it can be awaited.
        return new Promise((resolve, _) => { 
            // This is the event you’ll need to wait for before you can 
            // pass the video element to tf.fromPixels
            videoRef.onloadedmetadata = () => {
                // Prep Canvas
                const detection = document.getElementById('detection');
                const ctx = detection.getContext('2d');
                const imgWidth = videoRef.clientWidth; // clientWidth instead of width
                const imgHeight = videoRef.clientHeight;
                detection.width = imgWidth;
                detection.height = imgHeight;
                ctx.font = '16px sans-serif';
                ctx.textBaseline = 'top';
                // videoRef.play();
                // console.log(videoRef.clientHeight)
                // console.log(imgWidth, imgHeight);

                // The promise resolves with information you’ll need to 
                // pass along to the detect and draw loop
                resolve([ctx, imgHeight, imgWidth]);
            }
        })
    } else {
        alert('No webcam - sorry!')
    }
}

async function loadModel() {
    await tf.ready();
    // const modelPath = 'https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1';
    const modelPath = 'https://tfhub.dev/tensorflow/tfjs-model/ssdlite_mobilenet_v2/1/default/1'

    return await tf.loadGraphModel(modelPath, { fromTFHub: true })
}

async function doStuff() {
    try {
        const model = await loadModel();

        // For efficiency, you can capture the video element once and pass 
        // that reference into the places it’s needed.
        const mysteryVideo = document.getElementById('mystery');
        
        // Setting up the webcam should happen only once
        const camDetails = await setupWebcam(mysteryVideo);

        // stopBtn.addEventListener('click', stopWebcam(mysteryVideo));

        let front = true;
        flipBtn.addEventListener('click', async () => {
            // stopWebcam(mysteryVideo);

            front = !front; // Switch front boolean value
            constraints.video.facingMode = front ? "user" : "environment"; // Toggle camera facing mode
            console.log('flipped', constraints.video.facingMode)

            // const video = document.getElementById("mystery");
            const flippedCamDetails = await setupWebcam(mysteryVideo);
            performDetections(model, mysteryVideo, flippedCamDetails);
    // console.log(camDetails)
})

        // The performDetections method can loop forever when detecting the 
        // content in the webcam and drawing the boxes
        performDetections(model, mysteryVideo, camDetails);
    } catch (e) {
        // Don’t let the errors get swallowed up with all these awaits
        console.error(e)
    }
}

async function performDetections(model, videoRef, camDetails) {
    
    
    const [ctx, imgHeight, imgWidth] = camDetails;
    const video = document.getElementById('mystery');
    video.style.display = 'none';
    
    const myTensor = tf.browser.fromPixels(videoRef);
    // console.log(container.clientHeight, container.clientWidth)

    // SSD Mobilenet single batch
    // The input is expanded in rank to be a batch of one with 
    // the shape [1, height, width, 3]
    const readyfied = tf.expandDims(myTensor, 0);
    const results = await model.executeAsync(readyfied);

    // Get a clean tensor of top indices
    const detectionThreshold = 0.4;
    const iouThreshold = 0.5;
    const maxBoxes = 20;
    const prominentDetection = tf.topk(results[0]);
    const justBoxes = results[1].squeeze();
    const justValues = prominentDetection.values.squeeze();

    // Move results back to JavaScript in parallel
    const [maxIndices, scores, boxes] = await Promise.all([
        prominentDetection.indices.data(),
        justValues.array(),
        justBoxes.array(),
    ]);

    // https://arxiv.org/pdf/1704.04503.pdf, use Async to keep visuals
    const nmsDetections = await tf.image.nonMaxSuppressionWithScoreAsync(
        justBoxes, // shape [numBoxes, 4]
        justValues, // shape [numBoxes]
        maxBoxes, // Stop making boxes when this number is hit
        iouThreshold, // Allowed overlap value 0 to 1
        detectionThreshold, // Minimum detection score allowed
        1 // 0 is a normal NMS, 1 is max Soft-NMS
    );

    // Create a normal JavaScript array from the indices of the resulting 
    // high-scoring boxes
    const chosen = await nmsDetections.selectedIndices.data();
    // Mega Clean
    tf.dispose([
        results[0],
        results[1],
        // model, don't clean this one up for loops
        nmsDetections.selectedIndices,
        nmsDetections.selectedScores,
        prominentDetection.indices,
        prominentDetection.values,
        myTensor,
        readyfied,
        justBoxes,
        justValues,
    ]);

    // Clear everything each round
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

    let predsCount = {} // define data structure to store class count
    chosen.forEach((detection) => {
        ctx.strokeStyle = "#0F0";
        ctx.lineWidth = 4;

        // Draw under any existing content.
        // ctx.globalCompositeOperation = 'source-over';

        // Get the highest-scoring index from a previous topk call
        const detectedIndex = maxIndices[detection]

        // The classes are imported as an array to match the given result indices
        const detectedClass = CLASSES[detectedIndex];

        const detectedScore = scores[detection];
        const dBox = boxes[detection];

        // Log what is being boxed in the canvas so you can verify the results
        // console.log(detectedClass, detectedScore)

        // Count each detectedClass
        if (detectedClass in predsCount) {
            predsCount[detectedClass]++;
        } else {
            predsCount[detectedClass] = 1;
        };        

        // No negative values for start positions
        // const startY = dBox[0] > 0 ? dBox[0] * imgHeight: 0;
        // const startX = dBox[1] > 0 ? dBox[1] * imgWidth : 0;
        // const height = (dBox[2] - dBox[0]) * imgHeight;
        // const width = (dBox[3] - dBox[1]) * imgWidth;
        // console.log(startX, startY)
        // ctx.strokeRect(startX, startY, width, height);
        ctx.drawImage(videoRef, 0, 0, imgWidth, imgHeight);
        // console.log(imgHeight);
        const startY = dBox[0] > 0 ? dBox[0] * imgHeight: 0;
        const startX = dBox[1] > 0 ? dBox[1] * imgWidth : 0;
        const height = (dBox[2] - dBox[0]) * imgHeight;
        const width = (dBox[3] - dBox[1]) * imgWidth;
        ctx.strokeRect(startX, startY, width, height);
        // console.log(startX, startY, width, height);
        
        // Draw the label background

        // Draw over any existing content.
        // ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = '#0B0';
        // ctx.font = "16px sans-serif"; // Set the font and size to use on the labels
        ctx.textBaseline = "top"; // Set textBaseline as mentioned
        const textHeight = 16;
        const textPad = 4; // Add a little horizontal padding to be used in the fillRect render
        const label = `${detectedClass} ${Math.round(detectedScore * 100)}%`;
        const textWidth = ctx.measureText(label).width;
        // Draw the rectangle using the same startX and startY that were used to draw the bounding boxes
        ctx.fillRect(
            startX,
            startY,
            textWidth + textPad,
            textHeight + textPad
        )
        // Draw the text last to ensure it's on top
        ctx.fillStyle = '#000000'; // Change the fillStyle to be black for the text render
        ctx.fillText(label, startX, startY); // Draw the text;
        

        // console.log('Tensor Memory Status:', tf.memory().numTensors);        
    });

    // Loop through 'predsCount' data structure and log predicted classes 
    // and count
    for (const [predClass, count] of Object.entries(predsCount)) {
        console.log(predClass, count);        
    };

    
    // Loop forever 
    requestAnimationFrame(() => {
        performDetections(model, videoRef, camDetails);
    });
};

doStuff();

function stopWebcam (videoRef) {
    const stream = videoRef.srcObject;
    let tracks = stream.getTracks();
    for (let i = 0; i < tracks.length; i++) {
        let track = tracks[i];
        track.stop();
    }
    videoRef.srcObject = null;
}

// stopBtn.addEventListener('click', stopWebcam)

// let front = false;
// flipBtn.addEventListener('click', () => {
//     stopWebcam();

//     front = !front; // Switch front boolean value
//     constraints.video.facingMode = front ? "user" : "environment"; // Toggle camera facing mode
//     console.log('flipped', constraints.video.facingMode)

//     const video = document.getElementById("mystery");
//     const camDetails = setupWebcam(video);
//     performDetections(model, video, camDetails);
//     // console.log(camDetails)
// })