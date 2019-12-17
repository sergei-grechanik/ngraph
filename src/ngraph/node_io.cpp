//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/node_io.hpp"
#include "ngraph/node.hpp"

using namespace std;
using namespace ngraph;

NodeInput::NodeInput(const Node* node, size_t index)
    : m_node(node)
    , m_index(index)
{
}

Node* NodeInput::get_node() const
{
    return const_cast<Node*>(m_node);
}

size_t NodeInput::get_index() const
{
    return m_index;
}

const element::Type& NodeInput::get_element_type() const
{
    return m_node->get_input_element_type(m_index);
}

const Shape& NodeInput::get_shape() const
{
    return m_node->get_input_shape(m_index);
}

const PartialShape& NodeInput::get_partial_shape() const
{
    return m_node->get_input_partial_shape(m_index);
}

NodeOutput NodeInput::get_source_output() const
{
    auto& output_descriptor = m_node->m_inputs.at(m_index).get_output();
    return NodeOutput(output_descriptor.get_node(), output_descriptor.get_index());
}

descriptor::Tensor& NodeInput::get_tensor() const
{
    return m_node->m_inputs.at(m_index).get_output().get_tensor();
}

shared_ptr<descriptor::Tensor> NodeInput::get_tensor_ptr() const
{
    return m_node->m_inputs.at(m_index).get_output().get_tensor_ptr();
}

bool NodeInput::get_is_relevant_to_shapes() const
{
    return m_node->m_inputs.at(m_index).get_is_relevant_to_shape();
}

bool NodeInput::get_is_relevant_to_values() const
{
    return m_node->m_inputs.at(m_index).get_is_relevant_to_value();
}

void NodeInput::replace_source_output(const NodeOutput& new_source_output) const
{
    descriptor::Input di = m_node->m_inputs.at(m_index);
    di.replace_output(new_source_output.get_node_shared_ptr(), new_source_output.get_index());
}

bool NodeInput::operator==(const NodeInput& other) const
{
    return m_node == other.m_node && m_index == other.m_index;
}

bool NodeInput::operator!=(const NodeInput& other) const
{
    return !(*this == other);
}

bool NodeInput::operator<(const NodeInput& other) const
{
    return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
}

bool NodeInput::operator>(const NodeInput& other) const
{
    return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
}

bool NodeInput::operator<=(const NodeInput& other) const
{
    return !(*this > other);
}

bool NodeInput::operator>=(const NodeInput& other) const
{
    return !(*this < other);
}

NodeOutput::NodeOutput(const Node* node, size_t index)
    : m_node(const_cast<Node*>(node)->shared_from_this())
    , m_index(index)
{
}

NodeOutput::NodeOutput(const shared_ptr<const Node>& node, size_t index)
    : m_node(const_pointer_cast<Node>(node))
    , m_index(index)
{
}

NodeOutput NodeOutput::for_node(const shared_ptr<Node>& node)
{
    return NodeOutput(node, m_index);
}

Node* NodeOutput::get_node() const
{
    return m_node.get();
}

shared_ptr<Node> NodeOutput::get_node_shared_ptr() const
{
    return m_node;
}

shared_ptr<Node> NodeOutput::as_single_output_node(bool for_get_output_element) const
{
    return m_node->get_output_as_single_output_node(m_index, for_get_output_element);
}

size_t NodeOutput::get_index() const
{
    return m_index;
}

descriptor::Tensor& NodeOutput::get_tensor() const
{
    return m_node->m_outputs.at(m_index).get_tensor();
}

shared_ptr<descriptor::Tensor> NodeOutput::get_tensor_ptr() const
{
    return m_node->m_outputs.at(m_index).get_tensor_ptr();
}

const element::Type& NodeOutput::get_element_type() const
{
    return m_node->get_output_element_type(m_index);
}

const Shape& NodeOutput::get_shape() const
{
    return m_node->get_output_shape(m_index);
}

const PartialShape& NodeOutput::get_partial_shape() const
{
    return m_node->get_output_partial_shape(m_index);
}

set<ngraph::NodeInput> NodeOutput::get_target_inputs() const
{
    set<NodeInput> result;

    for (auto& input : m_node->m_outputs.at(m_index).get_inputs())
    {
        result.emplace(input->get_raw_pointer_node(), input->get_index());
    }

    return result;
}

void NodeOutput::remove_target_input(const NodeInput& target_input) const
{
    m_node->m_outputs.at(m_index).remove_input(
        &(target_input.get_node()->m_inputs.at(target_input.get_index())));
}

bool NodeOutput::operator==(const NodeOutput& other) const
{
    return m_node == other.m_node && m_index == other.m_index;
}

bool NodeOutput::operator!=(const NodeOutput& other) const
{
    return !(*this == other);
}

bool NodeOutput::operator<(const NodeOutput& other) const
{
    return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
}

bool NodeOutput::operator>(const NodeOutput& other) const
{
    return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
}

bool NodeOutput::operator<=(const NodeOutput& other) const
{
    return !(*this > other);
}

bool NodeOutput::operator>=(const NodeOutput& other) const
{
    return !(*this < other);
}
