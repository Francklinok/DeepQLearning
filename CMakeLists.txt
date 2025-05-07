cmake_minimum_required(VERSION 3.14)

project(DeepReinforcementLearning LANGUAGES CXX)

# Activer C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Rechargement à chaud pour MSVC
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
      "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,
      $<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,
      $<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Ajout de l'exécutable
# Listez uniquement les fichiers sources (.cpp) ici
add_executable(${PROJECT_NAME}
    DeepReinforcementLearning.cpp # Assurez-vous que votre fichier principal est dans src/
    # Ajoutez d'autres fichiers .cpp de votre projet si nécessaire
  "includes/core/ExploratoryStrategy/egreedy/DefaultEntropyCalculator.h" "includes/core/CompositeExploration/CompositeExploration .h")



# Inclure les répertoires de headers
# Dites au compilateur où trouver vos fichiers d'en-tête (.h ou .hpp)
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/includes # Ajoute le répertoire 'includes'
        ${CMAKE_CURRENT_SOURCE_DIR}/src      # Peut être utile si certains headers sont dans src
)

# TODO: Ajouter des modules ici (ex: core/, envs/, agents/)
# Si vous avez des fichiers .cpp dans ces sous-répertoires, ajoutez-les à add_executable
# ou utilisez add_subdirectory et liez les bibliothèques.
# Exemple si vous avez un fichier src/core/some_file.cpp :
# add_executable(${PROJECT_NAME}
#     src/DeepReinforcementLearning.cpp
#     src/core/some_file.cpp
# )

# (Facultatif) Ajouter les tests
# enable_testing()
# add_subdirectory(tests)

# (Facultatif) Installer le projet
# install(TARGETS ${PROJECT_NAME} DESTINATION bin)