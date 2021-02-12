
function randomDoubleInRange(min, max) {
  return Math.random() < 0.5 ? ((1-Math.random()) * (max-min) + min) : (Math.random() * (max-min) + min);
}

function shuffle(array) {
  return array.sort(() => Math.random() - 0.5);
}

// ----------------------------------------------------------------------------------------------------------------

class Perceptron1 {

  constructor(nInputs) {
    this.sop = 0;
    this.act = 0;
    this.input = [];
    this.weights = [];
    this.bias = 0.0;
    this.gradients = [];

    // Generate random weights
    for(let i = 0; i < nInputs; i++) {
      this.weights.push(randomDoubleInRange(0.0, 1.0));
    }
    // Generate random bias
    this.bias = randomDoubleInRange(0.0, 1.0);
  }

  /**
   *
   * @param input {array<number>} double
   * @returns {number} int
   */
  predict(input) {
    this.input = input;

    this.sop = 0;

    /**
     * i0 * w0 + i1 * w1 + ... + ix * wx
     */
    for(let i = 0; i < input.length; i++) {
      this.sop += input[i] * this.weights[i]
    }

    this.sop += this.bias;

    this.act = this.activateReLU(this.sop);

    return this.act;
  }

  /**
   * Fonction seuil
   * @param x {number} double
   * @returns {number} int
   */
  activateReLU(x) {
    if(x < 0) {
      return 0;
    }
    else {
      return x;
    }
  }

  /**
   *
   * @returns {number}
   */
  dActivate() {
    return this.sop > 0 ? 1 : 0;
  }

  /**
   *
   * @param output {number} double
   * @param expected {number} double
   * @returns {number} double
   */
  calcError(output, expected) {
    return expected - output;
  }

  /**
   *
   * @param delta {number} double
   */
  getGradient(delta) {
    this.gradients = [];
    for(let i = 0; i < this.input.length; i++) {
      this.gradients.push(this.input[i] * delta);
    }
    return this.gradients;
  }

  /**
   *
   * @param learningRate {number} double
   */
  updateWeights(learningRate) {
    for(let i = 0; i < this.weights.length; i++) {
      this.weights[i] -= this.gradients[i] * learningRate;
    }
  }
}

// ----------------------------------------------------------------------------------------------------------------

class DenseNet {

  /**
   *
   * @param nbInputs {number}
   * @param nbHiddenNeurons {number}
   * @param nbOutputs {number}
   */
  constructor(nbInputs, nbHiddenNeurons, nbOutputs) {
    // Perceptron1 arrays
    this.hiddenLayer1 = [];
    this.outputLayer = [];

    this.nbInputs = nbInputs;
    this.nbOutputs = nbOutputs;

    // Create inner neurons
    for(let i = 0; i < nbHiddenNeurons; i++) {
      this.hiddenLayer1.push(new Perceptron1(nbInputs));
    }

    // Create outputs
    for(let i = 0; i < nbOutputs; i++) {
      this.outputLayer.push(new Perceptron1(nbHiddenNeurons));
    }

  }

  /**
   *
   * @param input {number[]} double
   * @returns {number[]}
   */
  predict(input) {
    let hiddenOutput = [];

    // Put outputs in an array
    this.hiddenLayer1.map((perceptron) => {
      hiddenOutput.push(perceptron.predict(input));
    });

    const networkOutput = this.outputLayer[0].predict(hiddenOutput);

    return [networkOutput];
  }

  /**
   *
   * @param output {number[]} double
   * @param expected {number[]} double
   * @returns {number[]} double
   */
  calcNetworkError(output, expected) {
    let outputsErrors = [];
    for(let i = 0; i < output.length; i++) {
      outputsErrors.push(output[i] - expected[i]);
    }
    return outputsErrors;
  }

  /**
   *
   * @param input {number[]} double
   * @param expected {number[]} double
   * @param learningRate {number} double
   */
  train(input, expected, learningRate) {
    let pred = this.predict(input);
    let netError = this.calcNetworkError(pred, expected);

    let outDelta = [];
    for(let i = 0; i < this.outputLayer.length; i++) outDelta.push(0.0);

    // Pour chacune des sorties du réseau
    for(let j = 0; j < this.outputLayer.length; j++) {
      // Calculer terme à terme la somme des erreurs multipliées par la dérivée de chaque neurone caché
      for(let i = 0 ; i < this.hiddenLayer1.length; i++) {
        outDelta[j] += this.hiddenLayer1[i].dActivate() * netError[j];
      }
    }

    // Calcul du gradient en fonction du delta pour chaque neurone de sortie
    for(let i = 0; i < outDelta.length; i++) {
      this.outputLayer[i].getGradient(outDelta[i]);
    }

    // Créer un tableau d'erreur de taille hiddenLayer1 contenant des 0.0
    let l1Errors = [];
    for(let i = 0; i < this.hiddenLayer1.length; i++) l1Errors.push(0.0);

    // Obtenir l'erreur de chaque neurone caché
    let counter = 0;
    this.outputLayer.forEach((pOut) => {
      for(let i = 0; i < this.hiddenLayer1.length; i++) {
        l1Errors[i] += pOut.weights[i] * outDelta[counter];
      }
      counter++;
    });

    // Obtenir le delta de chaque neurone caché
    let hiddenDelta = [];
    for(let i = 0; i < this.hiddenLayer1.length; i++) {
      hiddenDelta.push(l1Errors[i] * this.hiddenLayer1[i].dActivate());
    }

    for(let i = 0; i < hiddenDelta.length; i++) {
      this.hiddenLayer1[i].getGradient(hiddenDelta[i]);
    }

    this.updateWeights(learningRate)
  }

  /**
   *
   * @param learningRate {number} double
   */
  updateWeights(learningRate) {
    this.outputLayer.map((perceptron) => {
      perceptron.updateWeights(learningRate);
    });
    this.hiddenLayer1.map((perceptron) => {
      perceptron.updateWeights(learningRate);
    });
  }
}

// ----------------------------------------------------------------------------------------------------------------

const dataset = [
  [[1.0, 0.0], [0.0]],
  [[0.0, 1.0], [0.0]],
  [[1.0, 1.0], [1.0]],
  [[0.0, 0.0], [0.0]],
];

const net = new DenseNet(2, 4, 1);

console.log("Before training")
dataset.map((data) => {
  console.log("For:", data[0], "expected:", data[1][0], "obtained:", net.predict(data[0]));
});

console.log("\nHidden Weights before train", net.hiddenLayer1[0].weights);
console.log("Output Weights before train", net.outputLayer[0].weights);
console.log("\n👾 Training...\n\n");
let datasetCopy = [...dataset];
for(let i = 0; i < 900; i++) { // 900 epoch
  shuffle(datasetCopy).map((data) => {
    net.train(data[0], data[1], 0.01);
  });
}
console.log("Hidden Weights after train", net.hiddenLayer1[0].weights);
console.log("Output Weights after train", net.outputLayer[0].weights, "\n\n");

console.log("After training")
dataset.map((data) => {
  console.log("For:", data[0], "expected:", data[1][0], "obtained:", Math.round(net.predict(data[0])[0]));
});
