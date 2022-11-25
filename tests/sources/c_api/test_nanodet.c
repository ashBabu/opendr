/*
 * Copyright 2020-2022 OpenDR European Project
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
 */
#include <check.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nanodet_c.h"
#include "opendr_utils.h"

START_TEST(model_creation_test) {
  // Create a nanodet libtorch model
  nanodet_model_t model;
  // Load a pretrained model
  load_nanodet_model("data/nanodet/optimized_model/nanodet_m.pth", "cpu", 320, 320, 0.35, &model);

  ck_assert_msg(model.net != 0, "net is NULL");

  // Release the resources
  free_nanodet_model(&model);

  // Check if memory steel exist
  ck_assert_msg(model.net, "net is NULL");
}
END_TEST

START_TEST(inference_creation_test) {
  // Create a nanodet model
  nanodet_model_t model;

  // Load a pretrained model
  load_nanodet_model("data/nanodet/optimized_model/nanodet_m.pth", "cpu", 320, 320, 0.35, &model);

  // Load an image and performance inference
  opendr_image_t image;
  load_image("data/nanodet/database/000000000036.jpg", &image);
  opendr_detection_target_list_t res = infer_nanodet(&image, &model);
  free_image(&image);

  ck_assert(res.size != 0);

  // Free the model resources
  free_detections(&res);
  free_nanodet_model(&model);
}
END_TEST

Suite *nanodet_suite(void) {
  Suite *s;
  TCase *tc_core;

  s = suite_create("Nanodet");
  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, model_creation_test);
  tcase_add_test(tc_core, inference_creation_test);
  suite_add_tcase(s, tc_core);

  return s;
}

int main() {
  int no_failed = 0;
  Suite *s;
  SRunner *runner;

  s = nanodet_suite();
  runner = srunner_create(s);

  srunner_run_all(runner, CK_NORMAL);
  no_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  return (no_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}