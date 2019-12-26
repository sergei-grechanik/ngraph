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

#include "ngraph/node.hpp"

using namespace ngraph;

NodeInput::NodeInput(const Node* node, size_t index)
    : m_node(node)
    , m_index(index)
{
}

NodeOutput NodeInput::get_source_output() const
{
    auto& output_descriptor = m_node->m_inputs.at(m_index).get_output();
    return NodeOutput(output_descriptor.get_node(), output_descriptor.get_index());
}

void NodeInput::replace_source_output(const NodeOutput& new_source_output) const
{
    const_cast<Node*>(m_node)->m_inputs.at(m_index).replace_output(
        new_source_output.get_node_shared_ptr(), new_source_output.get_index());
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

descriptor::Tensor& NodeInput::get_tensor() const
{
    return m_node->m_inputs.at(m_index).get_output().get_tensor();
}

std::shared_ptr<descriptor::Tensor> NodeInput::get_tensor_ptr() const
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

bool NodeInput::operator==(const NodeInput& other) const
{
    return m_node == other.m_node && m_index == other.m_index;
}

bool NodeInput::operator<(const NodeInput& other) const
{
    return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
}

bool NodeInput::operator>(const NodeInput& other) const
{
    return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
}
