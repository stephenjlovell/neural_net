//-----------------------------------------------------------------------------------
// ♛ GopherCheck ♛
// Copyright © 2015 Stephen J. Lovell
//-----------------------------------------------------------------------------------
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//-----------------------------------------------------------------------------------

package NeuralNet

import(
  // "fmt"
  "math"
  "math/rand"
)

const(
  ALPHA = 0.5
  ETA = 0.15
)

type Layer []*Neuron

func NewLayer(previous_size, size, next_size uint) *Layer {
  layer := make(Layer, size, size)
  for i := uint(0); i < size; i++ {
    layer[i] = NewNeuron(previous_size, next_size)
    layer[i].index = i
  }
  return &layer
}


type Neuron struct {
  connections []*Connection
  in chan float64

  incoming uint
  index uint

  output float64
  gradient float64

  eta float64 // net learning rate (0.0..1.0)
  alpha float64 // momentum
}

func (neuron *Neuron) Output() float64 {
  return neuron.output
}

func (neuron *Neuron) outputGradient(target float64) {
  neuron.gradient = (target - neuron.output) * neuron.activationDerivative(neuron.output)
}

func (neuron *Neuron) calcHiddenGradients(next_layer *Layer) {
  neuron.gradient = neuron.sumDOW(next_layer) * neuron.activationDerivative(neuron.output)
}

func (neuron *Neuron) sumDOW(next_layer *Layer) float64 {
  sum := 0.0
  for i, n := range *next_layer {
    sum += (neuron.connections[i].weight * n.gradient)
  }
  return sum
}

func (neuron *Neuron) updateInputWeights(prev_layer *Layer) {

  for _, n := range *prev_layer {
    old_delta_weight := n.connections[neuron.index].delta

    new_delta_weight := (neuron.eta * n.output * neuron.gradient) + 
      (neuron.alpha * old_delta_weight)

    n.connections[neuron.index].delta = new_delta_weight
  }

}

func (neuron *Neuron) FeedInitial(d float64) {

  neuron.output = neuron.activation(d)
  // fmt.Printf("%.3f -> %.3f\n", d, neuron.output)

  for _, conn := range neuron.connections {
    conn.out <- (neuron.output * conn.weight * conn.delta)
  }
}

func (neuron *Neuron) FeedForward() {

  sum := 0.0
  for i := uint(0); i < neuron.incoming; i++ {
    sum += <-neuron.in
  }

  neuron.output = neuron.activation(sum)
  // fmt.Printf("%.3f -> %.3f\n", sum, neuron.output)

  for _, conn := range neuron.connections {
    conn.out <- (neuron.output * conn.weight * conn.delta)
  }
}

func (neuron *Neuron) activation(sum float64) float64 {
  return (1.0 / (1.0 + math.Exp(-sum))) // sigmoid function
}

func (neuron *Neuron) activationDerivative(sum float64) float64 {
  return math.Exp(sum) / math.Pow(1.0 + math.Exp(sum), 2.0)
}



func NewNeuron(previous_size, next_size uint) *Neuron {
  neuron := &Neuron{
    eta: ETA,
    alpha: ALPHA,
    incoming: previous_size,
    in: make(chan float64, previous_size),
    connections: make([]*Connection, next_size, next_size),
  }
  for i := uint(0); i < next_size; i++ {
    neuron.connections[i] = NewConnection()
  }
  return neuron
}


type Connection struct {
  out chan float64
  weight float64
  delta float64
}

func NewConnection() *Connection {
  return &Connection{
    weight: connection_weight(),
    delta:  1.0,
  }
}

func connection_weight() float64 {
  return rand.Float64()
}









