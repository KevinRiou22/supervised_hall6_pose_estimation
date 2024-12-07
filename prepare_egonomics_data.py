
    task_actions = {1:["start_take_from_shelve", "end_take_from_shelve", "start_transport", "end_transport", "start_put_on_table", "end_put_on_table", "start_take_from_table", "end_take_from_table", "start_transport", "end_transport", "start_put_on_shelve", "end_put_on_shelve"],
                    2:["start_walk", "end_walk", "start_take_from_table", "end_take_from_table", "start_screw", "end_screw", "start_put_on_table", "end_put_on_table", "start_walk", "end_walk"],
                    3:["start_walk", "end_walk", "start_take_from_table", "end_take_from_table", "start_hammer", "end_hammer", "start_put_on_table", "end_put_on_table", "start_walk", "end_walk"]
                    }

    Task_ergo_label_variations ={"T1":{"E1":["crouch", "chest", "rotate", "rotate", "chest", "crouch"],
                                        "E2":["bend", "side", "rotate", "rotate", "side", "bend"],
                                        "E3":["crouch", "chest", "rotate", "rotate", "chest", "crouch"],
                                        "E4":["bend", "side", "rotate", "rotate", "side", "bend"],
                                        "E5":["upright", "chest", "rotate", "rotate", "chest", "upright"],
                                        "E6":["bend", "side", "rotate", "rotate", "side", "bend"],
                                        "E7":["upright", "chest", "rotate", "rotate", "chest", "upright"],
                                        "E8":["upright", "chest", "rotate", "rotate", "chest", "upright"],
                                        "E9":["upright", "side", "rotate", "rotate", "side", "upright"],
                                        "E10":["random", "shoulder", "rotate", "rotate", "shoulder", "random"],
                                        "E11":["random", "shoulder", "rotate", "rotate", "shoulder", "random"],
                                        "E12":["random", "shoulder", "rotate", "rotate", "shoulder", "random"],
                                        "E13":["random", "body_far", "rotate", "rotate", "body_far", "random"],
                                        "E14":["random", "body_far", "rotate", "rotate", "body_far", "random"],
                                        "E15":["random", "body_far", "rotate", "rotate", "body_far", "random"]
                                        },
                                "T2":{"E1":["-", "left_far", "left_close", "left_far", "-"],
                                        "E2":["-", "center_far", "center_close", "center_far", "-"],
                                        "E3":["-", "right_far", "right_close", "right_far", "-"],
                                        "E4":["-", "left_close", "left_far", "left_close", "-"],
                                        "E5":["-", "center_close", "center_far", "center_close", "-"],
                                        "E6":["-", "right_close", "right_far", "right_close", "-"]
                                        },
                                "T3":{"E1":["-", "left_far", "left_close", "left_far", "-"],
                                        "E2":["-", "center_far", "center_close", "center_far", "-"], #+"wrist broken"
                                        "E3":["-", "right_far", "right_close", "right_far", "-"],
                                        "E4":["-", "left_close", "left_far", "left_close", "-"],
                                        "E5":["-", "center_close", "center_far", "center_close", "-"],
                                        "E6":["-", "right_close", "right_far", "right_close", "-"]
                                        }
                                }

                                


