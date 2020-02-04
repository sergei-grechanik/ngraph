//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/node_input.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    std::ostream& operator<<(std::ostream& out, const Input<Node>& input)
    {
        return input.get_node()->write_description(out, 0) << ".input(" << input.get_index()
                                                           << "):" << input.get_element_type()
                                                           << input.get_partial_shape();
    }

    std::ostream& operator<<(std::ostream& out, const Input<const Node>& input)
    {
        return input.get_node()->write_description(out, 0) << ".input(" << input.get_index()
                                                           << "):" << input.get_element_type()
                                                           << input.get_partial_shape();
    }
}
