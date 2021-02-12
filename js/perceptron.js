
function randomDoubleInRange(min, max) {
  return Math.random() < 0.5 ? ((1-Math.random()) * (max-min) + min) : (Math.random() * (max-min) + min);
}

function shuffle(array) {
  return array.sort(() => Math.random() - 0.5);
}

// ----------------------------------------------------------------------------------------------------------------

function convertMachineToHooman(output) {
  return output === -1 ? "Ne mets pas de veste !!" : "Mets une veste !!";
}

// ----------------------------------------------------------------------------------------------------------------


class Perceptron {

  /**
   *
   * @param nInputs {number} Nombre de paramètres à traiter dans une analyse. Ici, 2 car : température et % d'humidité
   */
  constructor(nInputs) {
    this.weights = [];
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
    let sop = 0.0;

    /**
     * i0 * w0 + i1 * w1 + ... + ix * wx
     */
    for(let i = 0; i < input.length; i++) {
      sop += input[i] * this.weights[i]
    }

    sop += this.bias;

    return this.activateBinary(sop);
  }

  /**
   * Fonction seuil
   * @param x {number} double
   * @returns {number} int
   */
  activateBinary(x) {
    if(x > 0) {
      return 1;
    }
    else {
      return -1;
    }
  }

  /**
   *
   * @param input {array<number>} double
   * @param expected {number} double
   * @param learningRate {number} double
   */
  train(input, expected, learningRate = .4) {
    // Pour 20° et 5% de risque de pluie, quelle réponse donnes-tu ?
    const netOutput = this.predict(input);
    // Par rapport à la réponse que j'attendais, on calcule l'erreur
    // La différence entre prédiction et valeur attendue
    let netError = this.calcError(netOutput, expected);
    // Mettre à jour les poids
    for(let i = 0; i < this.weights.length; i++) {
      this.weights[i] = this.weights[i] + (netError * input[i]) * learningRate;
    }
    // Mettre à jour le biais
    this.bias += netError * learningRate;
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
}

// ----------------------------------------------------------------------------------------------------------------

let dataset = [
  // [input, output]

  [[20.0, 5.0], [-1]], // Ne pas mettre de veste
  [[18.0, 25.0], [-1]],
  [[24.0, 15.0], [-1]],
  [[22.0, 25.0], [-1]],
  [[21.0, 0.0], [-1]],
  [[21.0, 15.0], [-1]],
  [[25.0, 15.0], [-1]],
  [[25.0, 0.0], [-1]],

  [[5.0, 45.0], [1]], // Mettre une veste
  [[9.0, 5.0], [1]],
  [[12.0, 15.0], [1]],
  [[7.0, 25.0], [1]],
  [[8.0, 75.0], [1]],
  [[15.0, 75.0], [1]],
  [[9.0, 0.0], [1]],
];

const p = new Perceptron(dataset[0][0].length);

console.log("➡️ Results Before Training : ");
dataset.map((data) => {
  console.log("For:", data[0], "expected:", data[1][0], "obtained:", p.predict(data[0]));
});

console.log("\nBefore prediction Weights : ", p.weights);

console.log("👾 Training...");
let datasetCopy = [...dataset];
for(let i = 0; i < 500; i++) {
  shuffle(datasetCopy).map((data) => {
    p.train(data[0], data[1][0]);
  });
}

console.log("After prediction Weights : ", p.weights);

console.log("\n➡️ Results After Training : ");
dataset.map((data) => {
  console.log("For:", data[0], "expected:", data[1][0], "obtained:", p.predict(data[0]));
});
