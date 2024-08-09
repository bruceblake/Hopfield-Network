package main;

import java.util.ArrayList;

public class Main {

  public static void main(String[] args) {
    ArrayList<Neuron> neurons = new ArrayList<>();
    ArrayList<Connection> connections = new ArrayList<>();
    Neuron a = new Neuron(false);
    Neuron b = new Neuron(false);
    Neuron c = new Neuron(false);
    Neuron d = new Neuron(true);
    Connection ab = new Connection(a, b, -1);
    Connection bc = new Connection(b, c, 1);
    Connection bd = new Connection(b, d, -1);
    Connection ad = new Connection(a, d, 1);

    a.addConnection(ab);
    a.addConnection(ad);
    b.addConnection(ab);
    b.addConnection(bc);
    b.addConnection(bd);
    c.addConnection(bc);
    d.addConnection(bd);
    d.addConnection(ad);

    neurons.add(a);
    neurons.add(b);
    neurons.add(c);
    neurons.add(d);
    connections.add(ab);
    connections.add(bc);
    connections.add(bd);
    connections.add(ad);

    // {a: false} -{ -1 }> {b : false} -{ 1 }> {c : true}
    // -{1}> -{-1}> {d : true}

    int energy = 0;
    for (Connection connection : connections) {
      energy += connection.getHappinessOfEdge();
    }
    energy = 0 - energy;
    System.out.println("Starting Energy: " + energy);
    System.out.println("Solving...");
    Solve(neurons, connections, energy);

  }

  public static void Solve(ArrayList<Neuron> neurons, ArrayList<Connection> connections, int energy) {

    for (Connection connection : connections) {
      energy += connection.getHappinessOfEdge();
    }

    System.out.println("energy: " + energy);
    boolean changedNeuron = false;
    for (Neuron neuron : neurons) {
      int totalWeightedInput = 0;
      for (int i = 0; i < neuron.getNumberOfConnections(); i++) {
        Neuron otherNeuron = neuron.getConnection(i).getNeuronX() == neuron
            ? neuron.getConnection(i).getNeuronY()
            : neuron.getConnection(i).getNeuronX();
        totalWeightedInput += neuron.getConnection(i).getWeight() * otherNeuron.getStateAsInt();
      }
      if (totalWeightedInput > 0) {
        if (neuron.getState() == false) {
          neuron.setState(true);
          changedNeuron = true;
        }
      } else {
        if (neuron.getState() == true) {
          neuron.setState(false);
          changedNeuron = true;
        }
      }
    }

    if (changedNeuron) {
      Solve(neurons, connections, energy);
    } else {
      return;
    }
  }
}
