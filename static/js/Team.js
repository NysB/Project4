

const teamDataApiUrl = "https://nbadashboardproject.azurewebsites.net/teams";
const playerInfoApiUrl = "https://nbadashboardproject.azurewebsites.net/player_info";

let teamsData, playerInfoData;

function init() {
  // Fetch teams and player info from the APIs
  Promise.all([getTeams(), getPlayerInfo()])
    .then(([teams, playerInfo]) => {
      teamsData = teams;
      playerInfoData = playerInfo;
      
      populateDropdown(teams);
      const initialTeam = teams[0].Team;
      updateCharts(initialTeam);
    })
    .catch((error) => {
      console.log(error);
    });
}

function getTeams() {
  // Fetch team data from the API
  return fetch(teamDataApiUrl)
    .then((response) => response.json())
    .then((data) => {
      // Process the retrieved JSON data
      console.log(data);
      return data;
    });
}

function getPlayerInfo() {
  // Fetch player info from the API
  return fetch(playerInfoApiUrl)
    .then((response) => response.json())
    .then((data) => {
      // Process the retrieved JSON data
      console.log(data);
      return data;
    });
}

function populateDropdown(teams) {
  let dropdownMenu = document.querySelector("#selDataset");
  dropdownMenu.innerHTML = "";

  teams.forEach((team) => {
    let option = document.createElement("option");
    option.value = team.Team;
    option.textContent = team.Team;
    dropdownMenu.appendChild(option);
  });

  dropdownMenu.addEventListener("change", () => {
    const selectedTeam = dropdownMenu.value;
    updateCharts(selectedTeam);
  });
}

function updateCharts(selectedTeam) {
  const team = teamsData.find((team) => team.Team === selectedTeam);
  const playerInfo = playerInfoData.filter((info) => info.Current_team === selectedTeam);

  updateTeamBarChart(team);
  updatePlayerPieChart(playerInfo);
}

function updateTeamBarChart(team) {
  const apgScore = team.APG;
  const ppgScore = team.PPG;
  const rpgScore = team.RPG;

  let teamCategories = ["APG", "PPG", "RPG"];
  let teamScores = [apgScore, ppgScore, rpgScore];

  let dataPlot = [
    { x: teamCategories, y: teamScores, type: "bar" }
  ];

  let layout = {
    title: "Team Scores",
    xaxis: { title: "Category" },
    yaxis: { title: "Scores" }
  };

  Plotly.newPlot("teamBarChart", dataPlot, layout);
}

function updatePlayerPieChart(playerInfo) {
  let playerNames = playerInfo.map((info) => info.Player);
  let playerAges = playerInfo.map((info) => parseInt(info.Age));

  let data = {
    labels: playerNames,
    datasets: [
      {
        label: "Player Ages",
        data: playerAges,
        backgroundColor: generateRandomColors(playerNames.length), // Generate random colors for the bars
      },
    ],
  };

  function generateRandomColors(count) {
    let colors = [];
    for (let i = 0; i < count; i++) {
      let color = `rgba(${getRandomValue(0, 255)}, ${getRandomValue(0, 255)}, ${getRandomValue(0, 255)}, 0.8)`;
      colors.push(color);
    }
    return colors;
  }

  function getRandomValue(min, max) {
    return Math.floor(Math.random() * (max - min) + min);
  }

  const canvas = document.getElementById("playerPieChart");

  let chart = Chart.getChart(canvas);

  if (chart) {
    chart.data = data;
    chart.options.plugins.title.text = "Player Ages";
    chart.update();
  } else {
    chart = new Chart(canvas, {
      type: "bar",
      data: data,
      options: {
        plugins: {
          title: {
            display: true,
            text: "Player Ages",
          },
        },
      },
    });
  }
}

init();
