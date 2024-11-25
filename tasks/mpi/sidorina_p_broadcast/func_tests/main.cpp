#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sidorina_p_broadcast/include/ops_mpi.hpp"
#include "mpi/sidorina_p_broadcast/include/ops_mpi_m.hpp"

TEST(sidorina_p_broadcast_mpi, Test_arr3_term2) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>({1, 2, 3});
    terms = std::vector<int>({1, 2});

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    result.resize(array.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataRef->inputs_count.emplace_back(array.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataRef->inputs_count.emplace_back(terms.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataRef->outputs_count.emplace_back(result.size());
  }
  sidorina_p_broadcast_mpi::RefBroadcast testMpiTaskSequential(taskDataRef);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(m_result, result);
  }
}

TEST(sidorina_p_broadcast_mpi, Test_arr3_term3) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>({1, 2, 4});
    terms = std::vector<int>({1, 2, 3});

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    result.resize(array.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataRef->inputs_count.emplace_back(array.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataRef->inputs_count.emplace_back(terms.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataRef->outputs_count.emplace_back(result.size());
  }
  sidorina_p_broadcast_mpi::RefBroadcast testMpiTaskSequential(taskDataRef);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(m_result, result);
  }
}

TEST(sidorina_p_broadcast_mpi, Test_arr3_term6) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>({1, 2, 3});
    terms = std::vector<int>({1, 2, 3, 4, 5, 6});

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    result.resize(array.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataRef->inputs_count.emplace_back(array.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataRef->inputs_count.emplace_back(terms.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataRef->outputs_count.emplace_back(result.size());
  }
  sidorina_p_broadcast_mpi::RefBroadcast testMpiTaskSequential(taskDataRef);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(m_result, result);
  }
}

std::vector<int> randomVector(size_t size) {
  std::vector<int> v(size);
  std::random_device r;
  generate(v.begin(), v.end(), [&] { return r(); });
  return v;
}

TEST(sidorina_p_broadcast_mpi, Test_random) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>(0);
    array.resize(100, 0);
    for (unsigned long i = 0; i < array.size(); i++) {
      array[i] = round(rand() % 100);
    }
    terms = std::vector<int>(0);
    terms.resize(100, 0);
    for (unsigned long j = 0; j < terms.size(); j++) {
      terms[j] = round(rand() % 100);
    }

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    result.resize(array.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataRef->inputs_count.emplace_back(array.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataRef->inputs_count.emplace_back(terms.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataRef->outputs_count.emplace_back(result.size());
  }
  sidorina_p_broadcast_mpi::RefBroadcast testMpiTaskSequential(taskDataRef);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(m_result, result);
  }
}

TEST(sidorina_p_broadcast_mpi, Test_validation_array_1) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>();
    terms = std::vector<int>({1, 2, 3});

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}
TEST(sidorina_p_broadcast_mpi, Test_validation_terms_1) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>({1, 2, 3});
    terms = std::vector<int>();

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(sidorina_p_broadcast_mpi, Test_validation_1) {
  boost::mpi::communicator world;

  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> m_result;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    array = std::vector<int>();
    terms = std::vector<int>();

    m_result.resize(array.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataGlob->inputs_count.emplace_back(array.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskDataGlob->inputs_count.emplace_back(terms.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_result.data()));
    taskDataGlob->outputs_count.emplace_back(m_result.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}