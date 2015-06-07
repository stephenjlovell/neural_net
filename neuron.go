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
  "math/rand"
)

type Layer []*Neuron

func NewLayer(size, next_size uint) *Layer {
  layer := make(Layer, size, size)
  for i := uint(0); i < size; i++ {
    layer[i] = NewNeuron(next_size)
  }
  return &layer
}


type Neuron struct {
  in chan float64
  output float64
  connections []Connection
}

func (neuron *Neuron) FeedForward() {

}

func NewNeuron(size uint) *Neuron {
  neuron := &Neuron{
    connections: make([]Connection, size, size),
  }
  for i := uint(0); i < size; i++ {
    neuron.connections[i] = NewConnection()
  }
  return neuron
}


type Connection struct {
  out chan float64
  weight float64
  delta float64
}

func NewConnection() Connection {
  return Connection{
    out: make(chan float64),
    weight: connection_weight(),
    delta:  0,
  }
}

func connection_weight() float64 {
  return rand.Float64()
}









