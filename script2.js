let myCnnModel;
(async () => {
    //myCnnModel = await tf.loadLayersModel('Cnn/web_model_v4/model.json');
    myCnnModel = await tf.loadGraphModel('tfjs_model/model.json');
    console.log("Model loaded!");
})();

// Function to pipe 'pix' data to the loaded TF.js model
async function classifyDrawing(pixelData) {
    if (!myCnnModel) {
        console.log("Model not loaded yet.");
        return;
    }

    // Define original and target dimensions
    const W = 512;
    const H = 512;
    const TARGET_W = 28; // Assuming a 28x28 model
    const TARGET_H = 28;

    // Use tf.tidy() to get both logits and the predicted label
    const { logits, label } = tf.tidy(() => {
        // 1. Convert pix array to a 2D Tensor
        const inputTensor = tf.tensor(pixelData, [H, W], 'float32');

        // 2. Normalize (0..255 -> 0.0..1.0)
        const normalizedTensor = inputTensor.div(255.0);

        // 3. Reshape to 4D [1, H, W, 1]
        const reshapedTensor = normalizedTensor.reshape([1, H, W, 1]);

        // 4. Resize to target size [1, 28, 28, 1]
        const resizedTensor = tf.image.resizeBilinear(reshapedTensor, [TARGET_H, TARGET_W]);
        
        // 5. Run the prediction. This tensor *IS* the logits.
        const logitsTensor = myCnnModel.predict(resizedTensor);
        
        // 6. Get the inferred label by finding the index of the highest logit
        const labelTensor = logitsTensor.argMax(1);

        // 7. Get the raw data from the tensors
        return {
            logits: logitsTensor.dataSync(), // This is the full logits array
            label: labelTensor.dataSync()[0]  // This is the single highest index
        };
    });

    // Logits & label
    console.log("Logits (raw model output):", logits);
    console.log("Prediction (inferred label):", label);

    // Display the final prediction on the webpage
    const outputElement = document.getElementById('prediction-output');
    if (outputElement) {
        outputElement.innerText = `Prediction: ${label}`;
    }
}

// Pixel-accurate drawing
(() => {
    const canvas = document.getElementById('drawing_canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    // Fixed internal pixel grid
    const W = 512, H = 512; // stored resolution
    const ON = 255; // black pixel value (0..255)
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
            const v = 255 - pix[p]; // white bg, black ink
            data[i] = v; data[i+1] = v; data[i+2] = v; data[i+3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
    }

    function setPixel(x, y, v = ON) {
        if (x >= 0 && y >= 0 && x < W && y < H) pix[y * W + x] = v;
    }

    // Bresenham line
    function line(x0, y0, x1, y1, v = ON) {
        let dx = Math.abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        let dy = -Math.abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        let err = dx + dy;
        while (true) {
            setPixel(x0, y0, v);
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
            return;
        }
        const prev = history.pop();
        pix.set(prev);
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
        setPixel(last.x, last.y);
        render();
    };

    const move  = (e) => {
        if (!drawing) return;
        const p = eventToPixel(e);
        if (!p) return;
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
    //canvas.addEventListener('mouseleave', end);

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
    pix.fill(0);
    render();
})();
