package main;

import java.util.ArrayList;

public class Neuron {

  private boolean state = false;
  private ArrayList<Connection> connections;

  public Neuron(boolean state) {
    this.state = state;
    connections = new ArrayList<>();
  }

  public Connection getConnection(int index) {
    return connections.get(index);
  }

  public boolean getState() {
    return this.state;
  }

  public int getStateAsInt() {
    return this.state ? 1 : -1;
  }

  public void setState(boolean state) {
    this.state = state;
  }

  public void addConnection(Connection connection) {
    connections.add(connection);
  }

  public int getNumberOfConnections() {
    return this.connections.size();
  }

}
