/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::LSTM::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 7)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<LSTM>(new_args.at(0),
                             new_args.at(1),
                             new_args.at(2),
                             new_args.at(3),
                             new_args.at(4),
                             new_args.at(5),
                             new_args.at(6),
                             m_lstm_cell_shape);
}

op::LSTM::LSTM(std::shared_ptr<Node> param1_1,
               std::shared_ptr<Node> param1_2,
               std::shared_ptr<Node> param2_1,
               std::shared_ptr<Node> param2_2,
               std::shared_ptr<Node> bias1,
               std::shared_ptr<Node> bias2,
               std::shared_ptr<Node> lstm_cell_state,
               Shape lstm_cell_shape)
    : RequiresTensorViewArgs(
          "LSTM", {param1_1, param1_2, param2_1, param2_2, bias1, bias2, lstm_cell_state})
    , m_shape_input(param1_1->get_shape())
    , m_lstm_cell_shape(lstm_cell_shape)
{
    add_output(param1_1->get_element_type(), m_shape_input);
    add_output(param1_1->get_element_type(), m_lstm_cell_shape);
}

shared_ptr<Node> op::RNN::copy_with_new_args(const NodeVector& new_args) const
{
    return make_shared<RNN>(new_args, m_number_of_lstm_cells, m_lstm_output_shape);
}

op::RNN::RNN(const NodeVector& args, const int number_of_lstm_cells, Shape lstm_output_shape)
    : RequiresTensorViewArgs("RNN", args)
    , m_number_of_lstm_cells(number_of_lstm_cells)
    , m_lstm_output_shape(lstm_output_shape)
{
    for (size_t i = 0; i < number_of_lstm_cells; i++)
    {
        add_output(args[0]->get_element_type(), lstm_output_shape);
    }
}
