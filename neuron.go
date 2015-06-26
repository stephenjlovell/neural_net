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

import (
	// "fmt"
	"math"
	// "math/rand"
)

const (
	ALPHA = 0.5
	ETA   = 0.15
)

type Layer []*Neuron

func NewLayer(previous_size, size, next_size uint) *Layer {
	layer := make(Layer, size, size)
	for i := uint(0); i < size; i++ {
		layer[i] = NewNeuron(previous_size, next_size, i)
	}
	return &layer
}

type Neuron struct {
	connections []*Connection
	in          chan float64

	incoming uint
	index    uint

	output   float64
	gradient float64

	eta   float64 // net learning rate (0.0..1.0)
	alpha float64 // momentum (change in weight from previous training sample)
}

func NewNeuron(previous_size, next_size, index uint) *Neuron {
	neuron := &Neuron{
		index:       index,
		eta:         ETA,
		alpha:       ALPHA,
		incoming:    previous_size,
		in:          make(chan float64, previous_size),
		connections: make([]*Connection, next_size, next_size),
	}
	for i := uint(0); i < next_size; i++ {
		neuron.connections[i] = NewConnection()
	}
	return neuron
}

func (neuron *Neuron) Output() float64 {
	return neuron.output
}

func (neuron *Neuron) setOutputLayerGradients(target float64) {
	neuron.gradient = (target - neuron.output) * neuron.activationDerivative(neuron.output)
}

func (neuron *Neuron) setHiddenLayerGradients(next_layer *Layer) {
	neuron.gradient = neuron.sumDOW(next_layer) * neuron.activationDerivative(neuron.output)
}

func (neuron *Neuron) updateInputWeights(prev_layer *Layer) {
	for _, n := range *prev_layer {
		old_delta_weight := n.connections[neuron.index].delta_weight

		new_delta_weight := (neuron.eta * n.output * neuron.gradient) +
			(neuron.alpha * old_delta_weight)

		n.connections[neuron.index].delta_weight = new_delta_weight
		n.connections[neuron.index].weight += new_delta_weight
	}
}

func (neuron *Neuron) FeedInitial(d float64) {
	neuron.output = d
	for _, conn := range neuron.connections {
		conn.out <- (neuron.output * conn.weight * conn.delta_weight)
	}
}

func (neuron *Neuron) FeedForward() {
	go func() {
		for {
			sum := 0.0
			for i := uint(0); i < neuron.incoming; i++ {
				sum += <-neuron.in
			}
			neuron.output = neuron.activation(sum)

			for _, conn := range neuron.connections {
				conn.Send(neuron.output)
			}
		}
	}()
}

func (neuron *Neuron) activation(sum float64) float64 {
	// return (1.0 / (1.0 + math.Exp(-sum))) // sigmoid function
	return math.Tanh(sum) // hyperbolic tangent
}

func (neuron *Neuron) activationDerivative(sum float64) float64 {
	// return math.Exp(sum) / math.Pow(1.0 + math.Exp(sum), 2.0) // derivative of sigmoid function
	return 1.0 - (sum * sum)
}

func (neuron *Neuron) sumDOW(next_layer *Layer) float64 {
	sum := 0.0
	for i, n := range *next_layer {
		sum += (neuron.connections[i].weight * n.gradient)
	}
	return sum
}
