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
	"math"
	"sync"
)

const (
	SMOOTHING = 0.05
)

type Topology []uint
type Data []float64

type Net struct {
	layers []*Layer
	wg     sync.WaitGroup

	error                float64
	recent_avg_err       float64
	recent_avg_smoothing float64

	input_size  uint
	output_size uint

	first_run_complete bool
}

func NewNet(t Topology) *Net {
	layer_count := len(t)
	assert(layer_count >= 3, "Neural net must include at least one input, output, and hidden layer.")
	net := &Net{
		input_size:           t[0],
		output_size:          t[layer_count-1],
		layers:               make([]*Layer, layer_count, layer_count),
		recent_avg_smoothing: SMOOTHING,
	}
	net.layers[0] = NewLayer(net, 0, t[0], t[1]) // input layer
	for i := 1; i < layer_count-1; i++ {
		net.layers[i] = NewLayer(net, t[i-1], t[i], t[i+1]) // hidden layers
	}
	net.layers[layer_count-1] = NewLayer(net, t[layer_count-2], t[layer_count-1], 0) // output layers

	for i := layer_count - 1; i > 0; i-- {
		receivers, senders := net.layers[i], net.layers[i-1]
		for r, receiver := range *receivers {
			for s, sender := range *senders {
				sender.connections_out[r].out = receiver.in
				receiver.connections_in[s] = sender.connections_out[r]
				receiver.connections_in[s].backprop = sender.backprop
			}
		}
	}
	return net
}

func (net *Net) Start() {
	for i, layer := range net.layers[1:] { // begin listening for signals from input layer
		for _, neuron := range *layer {
			neuron.FeedForward()
			if i == 0 {
				neuron.setFirstHiddenLayerGradients()
			} else if i < len(net.layers)-2 {
				neuron.setHiddenLayerGradients()
			}
		}
	}
}

func (net *Net) FeedForward(input Data) {
	assert(net.input_size == uint(len(input)), "input data size mismatch")
	net.wg.Add(int(net.output_size))
	for i, neuron := range *net.layers[0] {
		neuron.FeedInitial(input[i])
	}
	net.wg.Wait()
}

func (net *Net) OutputLayer() Layer {
	return *net.layers[len(net.layers)-1]
}

func (net *Net) InputLayer() Layer {
	return *net.layers[0]
}

func (net *Net) Backpropegate(target Data) {
	// calculate net error
	net.error = 0.0
	out_layer := net.OutputLayer()
	for i, neuron := range out_layer {
		delta := target[i] - neuron.Output()
		net.error += (delta * delta)
	}
	net.error = math.Sqrt(net.error / float64(len(out_layer))) // Root Mean Square (RMS) error

	// exponential smoothing
	if !net.first_run_complete {
		net.recent_avg_err = net.error
	}
	net.recent_avg_err = ((1 - net.recent_avg_smoothing) * net.recent_avg_err) +
		(net.recent_avg_smoothing * net.error)

	// calculate gradients
	net.wg.Add(len(*net.layers[1]))
	for i, neuron := range out_layer {
		neuron.setOutputLayerGradients(target[i])
	}
	net.wg.Wait() // wait for all hidden layers to finish updating their gradients.
	// update connection weights for all layers
	for i := len(net.layers) - 1; i > 0; i-- {
		for _, neuron := range *net.layers[i] {
			neuron.updateInputWeights()
		}
	}
	net.first_run_complete = true
}

func (net *Net) GetResults() Data {
	output_layer := net.OutputLayer()
	l := len(output_layer)
	data := make(Data, l, l)
	for i, n := range output_layer {
		data[i] = n.output
	}
	return data
}
