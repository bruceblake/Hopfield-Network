package main;

import java.util.ArrayList;

public class Connection {
  Neuron connectedNeuronX;
  Neuron connectedNeuronY;
  int weight;
  boolean isExcitatory;
  boolean isHappy;

  public Connection(Neuron connectedNeuronX, Neuron connectedNeuronY, int weight) {
    this.connectedNeuronX = connectedNeuronX;
    this.connectedNeuronY = connectedNeuronY;
    this.weight = weight;
    this.isExcitatory = weight > 0;
  }

  public int getWeight() {
    return this.weight;
  }

  public Neuron getNeuronX() {
    return this.connectedNeuronX;
  }

  public Neuron getNeuronY() {
    return this.connectedNeuronY;
  }

  public boolean getHappinessOfConnection() {
    if (this.isExcitatory) {
      // Happy if both neurons have the same state (either both true or both false)
      isHappy = (getNeuronX().getState() == getNeuronY().getState());
    } else {
      // Happy if the neurons have different states (one true, one false)
      isHappy = (getNeuronX().getState() != getNeuronY().getState());
    }

    return isHappy;
  }

  public int getHappinessOfEdge() {
    return weight * connectedNeuronX.getStateAsInt() * connectedNeuronY.getStateAsInt();
  }

}
