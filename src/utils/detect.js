import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./renderBox";

const preprocess = (source, modelWidth, modelHeight) => {
  let xRatio, yRatio;

  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(source);
    const [h, w] = img.shape.slice(0, 2);
    const maxSize = Math.max(w, h);
    const imgPadded = img.pad([
      [0, maxSize - h],
      [0, maxSize - w],
      [0, 0],
    ]);

    xRatio = maxSize / w;
    yRatio = maxSize / h;

    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight])
      .div(255.0)
      .expandDims(0);
  });

  return [input, xRatio, yRatio];
};

export const detectImage = async (
  imgSource,
  model,
  classThreshold,
  canvasRef
) => {
  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3);

  tf.engine().startScope();
  const [input, xRatio, yRatio] = preprocess(
    imgSource,
    modelWidth,
    modelHeight
  );

  await model.net.executeAsync(input).then((res) => {
    const [boxes, scores, classes] = res.slice(0, 3);
    const boxes_data = boxes.dataSync();
    const scores_data = scores.dataSync();
    const classes_data = classes.dataSync();
    renderBoxes(
      canvasRef,
      classThreshold,
      boxes_data,
      scores_data,
      classes_data,
      [xRatio, yRatio]
    );
    tf.dispose(res);
  });

  tf.engine().endScope();
};

export const detectVideo = (vidSource, model, classThreshold, canvasRef) => {
  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3);

  const detectFrame = async () => {
    if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
      const ctx = canvasRef.getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      return;
    }

    tf.engine().startScope();
    const [input, xRatio, yRatio] = preprocess(
      vidSource,
      modelWidth,
      modelHeight
    );

    await model.net.executeAsync(input).then((res) => {
      const [boxes, scores, classes] = res.slice(0, 3);
      const boxes_data = boxes.dataSync();
      const scores_data = scores.dataSync();
      const classes_data = classes.dataSync();
      renderBoxes(
        canvasRef,
        classThreshold,
        boxes_data,
        scores_data,
        classes_data,
        [xRatio, yRatio]
      );
      tf.dispose(res);
    });

    requestAnimationFrame(detectFrame);
    tf.engine().endScope();
  };

  detectFrame();
};
