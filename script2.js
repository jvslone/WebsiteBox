let myCnnModel;
const CLASSES = ['Bird', 'Cat', 'Fish', 'Horse', 'Rabbit'];

function updateLogitsBar(logitsArray) {
    const container = document.getElementById('logits-readout');
    if (!container || !logitsArray) return;

    const logits = Array.from(logitsArray);
    if (logits.length !== CLASSES.length) return;

    const maxVal = Math.max(...logits);
    const maxIdx = logits.indexOf(maxVal);

    container.innerHTML = '';  // clear old content

    logits.forEach((val, i) => {
        const span = document.createElement('span');
        span.classList.add('logit-item');
        if (i === maxIdx) span.classList.add('max-logit');

        // show a few decimals so you can see changes
        span.textContent = `${CLASSES[i]}: ${val.toFixed(3)}`;

        container.appendChild(span);
    });
}

updateLogitsBar(new Array(CLASSES.length).fill(0.0));

(async () => {
    try {
        myCnnModel = await tf.loadGraphModel('tfjs_model/model.json');
        console.log("Model loaded!");
        const outputElement = document.getElementById('prediction-output');
        if (outputElement) {
            outputElement.innerText = "Model loaded. Draw a bird, cat, fish, horse, or rabbit!";
        }
    } catch (e) {
        console.error("Error loading model:", e);
        const outputElement = document.getElementById('prediction-output');
        if (outputElement) {
            outputElement.innerText = "Error loading model. See console.";
        }
    }
})();

async function classifyDrawing(pixelData) {
    if (!myCnnModel) {
        console.log("Model not loaded yet.");
        return;
    }

    const W = 512;
    const H = 512;
    const TARGET_W = 28;
    const TARGET_H = 28;

    let logits, label;

    try {
        // 1. Make input tensor and check for empty canvas
        const inputTensor = tf.tensor(pixelData, [H, W], 'float32').div(255.0);

        const maxVal = (await inputTensor.max().data())[0];
        if (maxVal === 0 || isNaN(maxVal)) {
            console.log("Empty canvas detected.");
            logits = [0, 0, 0, 0, 0];
            label = 0; // default to Bird
            inputTensor.dispose();
        } else {
            // 2. Find non-zero coordinates ASYNCHRONOUSLY
            const nonZeroMask = inputTensor.greater(0);
            const nonZeroCoordsT = await tf.whereAsync(nonZeroMask); // shape [N, 2]
            const nonZeroCoords = await nonZeroCoordsT.array();      // JS array [[y,x], ...]

            nonZeroMask.dispose();
            nonZeroCoordsT.dispose();

            if (nonZeroCoords.length === 0) {
                // Fallback: treat as empty just in case
                console.log("No non-zero pixels found.");
                logits = [0, 0, 0, 0, 0];
                label = 0;
                inputTensor.dispose();
            } else {
                // 3. Compute bounding box in JS
                let yMin = H, yMax = 0, xMin = W, xMax = 0;
                for (const [y, x] of nonZeroCoords) {
                    if (y < yMin) yMin = y;
                    if (y > yMax) yMax = y;
                    if (x < xMin) xMin = x;
                    if (x > xMax) xMax = x;
                }

                const boxPad = 20;
                const yMinPad = Math.max(0, yMin - boxPad);
                const yMaxPad = Math.min(H - 1, yMax + boxPad);
                const xMinPad = Math.max(0, xMin - boxPad);
                const xMaxPad = Math.min(W - 1, xMax + boxPad);

                let height = yMaxPad - yMinPad + 1;
                let width  = xMaxPad - xMinPad + 1;
                if (height <= 0) height = 1;
                if (width  <= 0) width  = 1;

                // 4. Crop, pad to square, resize, and predict
                const croppedTensor = inputTensor.slice([yMinPad, xMinPad], [height, width]);

                const maxDim = Math.max(height, width);
                const padH = Math.floor((maxDim - height) / 2);
                const padW = Math.floor((maxDim - width) / 2);

                const paddedSquare = croppedTensor.pad(
                    [[padH, maxDim - height - padH], [padW, maxDim - width - padW]],
                    0
                );

                const reshaped = paddedSquare.reshape([1, maxDim, maxDim, 1]);
                const resizedTensor = tf.image.resizeBilinear(reshaped, [TARGET_H, TARGET_W]);

                const logitsT = myCnnModel.predict(resizedTensor);
                const labelT  = logitsT.argMax(1);

                logits = await logitsT.data();
                label  = (await labelT.data())[0];

                // Clean up
                logitsT.dispose();
                labelT.dispose();
                resizedTensor.dispose();
                reshaped.dispose();
                paddedSquare.dispose();
                croppedTensor.dispose();
                inputTensor.dispose();
            }
        }
    } catch (e) {
        console.error("Error during classifyDrawing:", e);
        logits = [0, 0, 0, 0, 0];
        label = 0;
    }

    console.log("Logits (raw model output):", logits);
    const predictionName = CLASSES[label];
    console.log("Prediction (inferred label):", label, predictionName);

    // NEW: update top-bar logits display
    updateLogitsBar(logits);

    const outputElement = document.getElementById('prediction-output');
    if (outputElement) {
        outputElement.innerText = `Prediction: ${predictionName}`;
    }
}


// Pixel-accurate drawing
(() => {
    // ... (The rest of this file is unchanged) ...
    const canvas = document.getElementById('drawing_canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    // Fixed internal pixel grid
    const W = 512, H = 512; // stored resolution
    const ON = 255; // black pixel value (0..255)
    const OFF = 0; // white pixel value
    const BRUSH_SIZE = 5; // Draw a 20x20 square brush
    const pix = new Uint8Array(W * H);

    // History stack for undo (each entry is a copy of pix before a stroke)
    const history = [];
    const MAX_HISTORY = 50; // optional cap

    // Internal canvas pixels = logical grid
    canvas.width = W;
    canvas.height = H;

    let imgData = ctx.createImageData(W, H);

    function render() {
        const data = imgData.data;
        for (let p = 0, i = 0; p < pix.length; p++, i += 4) {
            const v = 255 - pix[p]; // white bg (0->255), black ink (255->0)
            data[i] = v; data[i+1] = v; data[i+2] = v; data[i+3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
    }

    // UPDATED setPixel function for a thick brush
    function setPixel(x, y, v = ON) {
        // safety check for NaN
        if (isNaN(x) || isNaN(y)) {
            console.error("Invalid coordinates passed to setPixel:", x, y);
            return;
        }
        for (let i = -BRUSH_SIZE; i < BRUSH_SIZE; i++) {
            for (let j = -BRUSH_SIZE; j < BRUSH_SIZE; j++) {
                const px = x + i;
                const py = y + j;
                // Check bounds
                if (px >= 0 && py >= 0 && px < W && py < H) {
                    pix[py * W + px] = v;
                }
            }
        }
    }

    // Bresenham line (no change needed, it calls setPixel)
    function line(x0, y0, x1, y1, v = ON) {
        // safety check for NaN
        if (isNaN(x0) || isNaN(y0) || isNaN(x1) || isNaN(y1)) {
            console.error("Invalid coordinates passed to line:", x0, y0, x1, y1);
            return;
        }
        let dx = Math.abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        let dy = -Math.abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        let err = dx + dy;
        while (true) {
            setPixel(x0, y0, v); // This will now draw a 20x20 block
            if (x0 === x1 && y0 === y1) break;
            const e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }

    // Map mouse/touch point to pixel coords
    function eventToPixel(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = (e.clientX ?? e.touches?.[0]?.clientX);
        const clientY = (e.clientY ?? e.touches?.[0]?.clientY);
        if (clientX == null || clientY == null) return null;

        const cx = clientX - rect.left;
        const cy = clientY - rect.top;
        const sx = canvas.width / rect.width;
        const sy = canvas.height / rect.height;
        return { x: Math.floor(cx * sx), y: Math.floor(cy * sy) };
    }

    let drawing = false;
    let last = null;

    function pushHistorySnapshot() {
        // Save a copy of the current pix before starting a new stroke
        history.push(pix.slice());
        if (history.length > MAX_HISTORY) {
            history.shift(); // drop oldest
        }
    }

    function undoLastStroke() {
        if (history.length === 0) {
            console.log("Nothing to undo.");
            // If history is empty, clear the canvas
            pix.fill(OFF);
        } else {
            const prev = history.pop();
            pix.set(prev);
        }
        render();
        classifyDrawing(pix); // Re-classify after undo
    }

    const start = (e) => {
        const p = eventToPixel(e);
        if (!p) return;

        // New stroke: save state for undo
        pushHistorySnapshot();

        drawing = true;
        last = p;
        setPixel(last.x, last.y); // This was the typo
        render();
    };

    const move  = (e) => {
        if (!drawing) return;
        const p = eventToPixel(e);
        // --- NEW FIX: Check if p is null ---
        if (!p || !last) {
            drawing = false;
            last = null;
            return;
        };
        // --- END OF NEW FIX ---
        line(last.x, last.y, p.x, p.y);
        last = p;
        render();
    };

    const end   = () => {
        if (!drawing) return;
        drawing = false;
        last = null;
        classifyDrawing(pix); // Run prediction when the user lifts the mouse
    };

    // Mouse
    canvas.addEventListener('mousedown', start);
    canvas.addEventListener('mousemove', move);
    canvas.addEventListener('mouseup', end);
    canvas.addEventListener('mouseleave', end); // Add mouseleave to stop drawing

    // Touch
    canvas.addEventListener('touchstart', (e) => { e.preventDefault(); start(e); }, { passive: false });
    canvas.addEventListener('touchmove',  (e) => { e.preventDefault(); move(e);  }, { passive: false });
    canvas.addEventListener('touchend', end);

    // Undo button
    const undoButton = document.getElementById('undo-button');
    if (undoButton) {
        undoButton.addEventListener('click', () => {
            undoLastStroke();
        });
    }

    // Ctrl+Z / Cmd+Z for undo
    document.addEventListener('keydown', (e) => {
        const isUndoKey = (e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey);
        if (isUndoKey) {
            e.preventDefault();
            undoLastStroke();
        }
    });

    // Init white canvas
    pix.fill(OFF);
    render();
})();

console.log("Script loaded. Ready to draw and classify!");