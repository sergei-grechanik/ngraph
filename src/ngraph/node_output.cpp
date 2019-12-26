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

NodeOutput::NodeOutput(const Node* node, size_t index)
    : m_node(const_cast<Node*>(node)->shared_from_this())
    , m_index(index)
{
}

NodeOutput::NodeOutput(const std::shared_ptr<const Node>& node, size_t index)
    : m_node(std::const_pointer_cast<Node>(node))
    , m_index(index)
{
}

std::set<NodeInput> NodeOutput::get_target_inputs() const
{
    std::set<NodeInput> result;

    for (auto& input : m_node->m_outputs.at(m_index).get_inputs())
    {
        result.emplace(input->get_raw_pointer_node(), input->get_index());
    }

    return result;
}

Node* NodeOutput::get_node() const
{
    return m_node.get();
}

void NodeOutput::remove_target_input(const NodeInput& target_input) const
{
    m_node->m_outputs.at(m_index).remove_input(
        &(target_input.get_node()->m_inputs.at(target_input.get_index())));
}

std::shared_ptr<Node> NodeOutput::as_single_output_node(bool for_get_output_element) const
{
    return m_node->get_output_as_single_output_node(m_index, for_get_output_element);
}

descriptor::Tensor& NodeOutput::get_tensor() const
{
    return m_node->m_outputs.at(m_index).get_tensor();
}

std::shared_ptr<descriptor::Tensor> NodeOutput::get_tensor_ptr() const
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

bool NodeOutput::operator==(const NodeOutput& other) const
{
    return m_node == other.m_node && m_index == other.m_index;
}

bool NodeOutput::operator<(const NodeOutput& other) const
{
    return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
}

bool NodeOutput::operator>(const NodeOutput& other) const
{
    return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
}